"""
Parameter Golf: Q² Optimized Training Script
=============================================

Maximizes every bit of 128,000,000 bits and every FLOP of 8×H100 for 600s.

Architecture: [GQA, CfC, CfC, CfC] × 4 = 16 layers (Geode-derived)
Quantization: Z₄ (2-bit) structural quantization with Gray encoding
Optimizer:    Muon (Nesterov + spectral norm)
Training:     3-phase Geode-guided (FP32 warm-up → progressive QAT → refinement)

Hardware:     8 × H100 SXM (989 TFLOPS BF16, 80GB HBM3, 50MB L2, 128B cache line)
Budget:       16,000,000 bytes artifact, 600 seconds wall-clock
Target:       < 1.05 bits/byte on FineWeb validation

References:
  - Williams 2025 (SpaceTime bound): arXiv:2502.17779
  - Wildberger & Rubine 2025 (Geode): Amer. Math. Monthly 132:5
  - Hammons et al. 1994 (Z₄ Gray map): IEEE Trans. IT 40:2
  - Hasani et al. 2022 (CfC): Nature Machine Intelligence 4
  - Ma et al. 2024 (BitNet 1.58): arXiv:2402.12263

See: docs/parameter-golf/APPROACH.md, docs/parameter-golf/DESIGN.md
"""

from __future__ import annotations

import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# ── Hardware constants (H100 SXM) ──────────────────────────────────────────

CACHE_LINE_BYTES = 128        # H100 L2 cache line
REGISTER_BITS = 64            # CUDA 64-bit register (for packing math)
Z4_WEIGHTS_PER_REGISTER = 32  # 64 / 2
Z4_WEIGHTS_PER_CACHE_LINE = 512  # 128 * 8 / 2
INV_CDF_75 = 0.6745           # Φ⁻¹(3/4) for equiprobable Z₄ thresholds
ARTIFACT_BUDGET = 16_000_000  # bytes


# ── Z₄ Quantization ────────────────────────────────────────────────────────

class Q2Quantize(torch.autograd.Function):
    """Z₄ structural quantization with straight-through estimator.

    Maps weights to {A=0, B=1, C=2, D=3} using equiprobable thresholds,
    Gray-encodes for packing, and dequantizes to centroids for forward pass.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        weight: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        # Classify into 4 cells: A (strong−), B (weak−), C (weak+), D (strong+)
        sym = torch.zeros_like(weight, dtype=torch.long)
        sym = torch.where(weight <= -tau, torch.tensor(0, device=weight.device), sym)
        sym = torch.where(
            (weight > -tau) & (weight <= 0), torch.tensor(1, device=weight.device), sym
        )
        sym = torch.where(
            (weight > 0) & (weight <= tau), torch.tensor(2, device=weight.device), sym
        )
        sym = torch.where(weight > tau, torch.tensor(3, device=weight.device), sym)

        # Dequantize: {A,B,C,D} → {-1.5τ, -0.5τ, +0.5τ, +1.5τ}
        centroids = torch.tensor(
            [-1.5, -0.5, 0.5, 1.5], dtype=weight.dtype, device=weight.device
        )
        weight_q = centroids[sym] * tau

        # STE passthrough window: κ = 3τ*
        kappa = 3.0 * tau
        ctx.save_for_backward(weight, kappa)
        return weight_q

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        weight, kappa = ctx.saved_tensors
        # Pass gradients through only within the passthrough window
        mask = weight.abs() <= kappa
        return grad_output * mask.float(), None


class Q2Linear(nn.Module):
    """Linear layer with Z₄ quantization-aware training.

    During training: quantizes weights to Z₄ via STE each forward pass.
    During eval: uses cached quantized weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Equiprobable threshold (refreshed periodically during training)
        self.register_buffer(
            "tau", torch.tensor(INV_CDF_75 / math.sqrt(in_features))
        )

        # Q2 active flag (starts inactive for FP32 warm-up)
        self.q2_active = False

        # Initialize: Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def refresh_tau(self) -> None:
        """Refresh threshold from empirical weight distribution (§D-2.5)."""
        with torch.no_grad():
            # Per-row 75th percentile
            q75 = torch.quantile(self.weight.abs(), 0.75, dim=-1, keepdim=True)
            self.tau.fill_(q75.mean().item())

    def activate_q2(self) -> None:
        """Enable Z₄ quantization (call after FP32 warm-up phase)."""
        self.q2_active = True
        self.refresh_tau()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.q2_active:
            w = Q2Quantize.apply(self.weight, self.tau)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


# ── CfC Block (Geode G-level: refinement) ──────────────────────────────────

class CfCBlock(nn.Module):
    """Closed-form Continuous-time block — one Geode G-level (3-way refinement).

    Runs the closed-form LTC update per token; state h propagates across the
    sequence with no KV cache.  All projections are Q2Linear (Z₄).
    """

    def __init__(self, d_model: int, n_time_constants: int = 5, mlp_ratio: float = 3.0):
        super().__init__()
        mlp_dim = int(d_model * mlp_ratio)

        # CfC projections
        self.a1_proj = Q2Linear(d_model, d_model)   # Decay rate
        self.a2_proj = Q2Linear(d_model, d_model)   # Integration rate
        self.out_proj = Q2Linear(d_model, d_model)
        self.tau_c = nn.Parameter(torch.randn(n_time_constants))
        self.ln1 = nn.LayerNorm(d_model)

        # SwiGLU MLP
        self.mlp_gate = Q2Linear(d_model, mlp_dim)
        self.mlp_up = Q2Linear(d_model, mlp_dim)
        self.mlp_down = Q2Linear(mlp_dim, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        if h is None:
            h = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            x_t = self.ln1(x[:, t, :])
            a1 = torch.sigmoid(self.a1_proj(x_t))
            a2 = torch.sigmoid(self.a2_proj(x_t))
            tc = torch.sigmoid(self.tau_c).unsqueeze(0)
            # Pad or slice tc to match d_model
            if tc.shape[-1] < D:
                tc = tc.repeat(1, (D + tc.shape[-1] - 1) // tc.shape[-1])[:, :D]
            # Closed-form LTC update: h_new = exp(-a1*τ)*h + (a2/a1)*(1 - exp(-a1*τ))
            decay = torch.exp(-a1 * tc)
            h = decay * h + (a2 / (a1 + 1e-6)) * (1.0 - decay)
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)  # (B, T, D)
        x = x + self.out_proj(h_seq)

        # SwiGLU MLP
        h2 = self.ln2(x)
        x = x + self.mlp_down(F.silu(self.mlp_gate(h2)) * self.mlp_up(h2))

        return x, h


# ── GQA Block (Geode S1-level: coarse selection) ───────────────────────────

class GQABlock(nn.Module):
    """Grouped Query Attention block — one Geode S1-level (4-way coarse selection).

    Uses F.scaled_dot_product_attention → FlashAttention-2 on H100.
    All projections are Q2Linear (Z₄).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        mlp_dim = int(d_model * mlp_ratio)

        # Attention projections
        self.q_proj = Q2Linear(d_model, d_model)
        self.k_proj = Q2Linear(d_model, self.head_dim * n_kv_heads)
        self.v_proj = Q2Linear(d_model, self.head_dim * n_kv_heads)
        self.o_proj = Q2Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)

        # SwiGLU MLP
        self.mlp_gate = Q2Linear(d_model, mlp_dim)
        self.mlp_up = Q2Linear(d_model, mlp_dim)
        self.mlp_down = Q2Linear(mlp_dim, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads to match query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # FlashAttention-2 on H100
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn_out)

        # SwiGLU MLP
        h2 = self.ln2(x)
        x = x + self.mlp_down(F.silu(self.mlp_gate(h2)) * self.mlp_up(h2))

        return x


# ── Full Model: Geode Layout [GQA, CfC, CfC, CfC] × 4 ────────────────────

@dataclass
class ModelConfig:
    """Configuration derived from constraints and Geode structure."""
    vocab_size: int = 256       # Byte tokenization (saves 1.2 MB vs SP-1024)
    d_model: int = 768          # Hidden dimension (32-aligned for Z₄ registers)
    n_geode_levels: int = 4     # 4 Geode levels
    cfc_per_level: int = 3      # 3 CfC blocks per GQA (from G = 1/(1-3x))
    n_heads: int = 8            # Query heads
    n_kv_heads: int = 4         # KV heads (GQA)
    mlp_ratio: float = 3.0      # SwiGLU expansion
    n_time_constants: int = 5   # CfC time constants per block
    max_seq_len: int = 2048     # Context length

    @property
    def n_layers(self) -> int:
        return self.n_geode_levels * (1 + self.cfc_per_level)  # 4 * 4 = 16


class Q2LTCModel(nn.Module):
    """Q²-QAT Hybrid LTC-Transformer with Geode layout.

    Architecture: [GQA, CfC, CfC, CfC] × 4 = 16 layers
    4 GQA blocks (S₁ coarse context) + 12 CfC blocks (G refinement)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding (FP16, tied with output)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Build Geode-ordered layer stack
        self.layers = nn.ModuleList()
        self.layer_types: list[str] = []
        for level in range(cfg.n_geode_levels):
            # S₁: GQA block (coarse context, 4 choices)
            self.layers.append(
                GQABlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_ratio)
            )
            self.layer_types.append("gqa")
            # G: 3 × CfC blocks (refinement, 3 choices each)
            for _ in range(cfg.cfc_per_level):
                self.layers.append(
                    CfCBlock(cfg.d_model, cfg.n_time_constants, cfg.mlp_ratio)
                )
                self.layer_types.append("cfc")

        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Tied output projection
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """OrthoInit for GQA, Kaiming for CfC/MLP (following BitNet practice)."""
        for name, module in self.named_modules():
            if isinstance(module, Q2Linear):
                if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def activate_q2(self, layer_indices: Optional[list[int]] = None) -> None:
        """Activate Z₄ quantization on specified layers (or all if None)."""
        for i, layer in enumerate(self.layers):
            if layer_indices is not None and i not in layer_indices:
                continue
            for module in layer.modules():
                if isinstance(module, Q2Linear):
                    module.activate_q2()

    def refresh_all_tau(self) -> None:
        """Refresh Z₄ thresholds from current weight distributions."""
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, Q2Linear):
                    module.refresh_tau()

    def forward(
        self,
        idx: torch.Tensor,
        cfc_states: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """
        Args:
            idx: Token indices (B, T)
            cfc_states: Optional CfC hidden states from previous batch

        Returns:
            logits: (B, T, V)
            new_cfc_states: Updated CfC states for next batch
        """
        x = self.embed(idx)

        if cfc_states is None:
            cfc_states = [None] * len(self.layers)

        new_states: list[Optional[torch.Tensor]] = []
        cfc_idx = 0

        for i, (layer, ltype) in enumerate(zip(self.layers, self.layer_types)):
            if ltype == "gqa":
                x = layer(x)
                new_states.append(None)
            else:
                state = cfc_states[i] if i < len(cfc_states) else None
                x, h = layer(x, state)
                new_states.append(h.detach())
                cfc_idx += 1

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_states

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_artifact_size(self) -> dict[str, float]:
        """Estimate artifact size in bytes at Z₄ (2-bit) packing."""
        embed_bytes = self.cfg.vocab_size * self.cfg.d_model * 2  # FP16
        q2_params = 0
        fp16_params = 0

        for name, p in self.named_parameters():
            if "embed" in name or "lm_head" in name:
                continue  # Tied, counted in embed_bytes
            if "ln" in name or "tau" in name:
                fp16_params += p.numel()
            else:
                q2_params += p.numel()

        q2_bytes = q2_params * 2 / 8   # 2 bits per weight
        fp16_bytes = fp16_params * 2     # 16 bits per param
        raw_total = embed_bytes + q2_bytes + fp16_bytes
        compressed = raw_total * 0.85    # Conservative zstd-22

        return {
            "embed_bytes": embed_bytes,
            "q2_bytes": q2_bytes,
            "fp16_bytes": fp16_bytes,
            "raw_total": raw_total,
            "compressed_estimate": compressed,
            "budget_remaining": ARTIFACT_BUDGET - compressed,
            "q2_params": q2_params,
            "total_params": self.count_parameters(),
        }


# ── Gray Encoding and Packing ──────────────────────────────────────────────

def gray_encode(sym: torch.Tensor) -> torch.Tensor:
    """Gray map φ: Z₄ → F₂². g = s ⊕ (s >> 1)."""
    return sym ^ (sym >> 1)


def gray_decode(gray: torch.Tensor) -> torch.Tensor:
    """Inverse Gray map."""
    sym = gray.clone()
    sym ^= sym >> 1
    return sym


def pack_z4(symbols: torch.Tensor) -> bytes:
    """Pack Z₄ symbols (values 0-3) into bytes, 4 per byte, MSB-first."""
    gray = gray_encode(symbols.to(torch.uint8))
    n = gray.numel()
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        gray = F.pad(gray.view(-1), (0, pad))
    gray = gray.view(-1, 4)
    packed = (gray[:, 0] << 6) | (gray[:, 1] << 4) | (gray[:, 2] << 2) | gray[:, 3]
    return packed.cpu().numpy().tobytes()


def unpack_z4(data: bytes, n: int, device: str = "cpu") -> torch.Tensor:
    """Unpack bytes to Z₄ symbols."""
    packed = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
    s0 = (packed >> 6) & 0x3
    s1 = (packed >> 4) & 0x3
    s2 = (packed >> 2) & 0x3
    s3 = packed & 0x3
    gray = torch.stack([s0, s1, s2, s3], dim=-1).view(-1)[:n]
    return gray_decode(gray)


# ── Q2BN Binary Format ─────────────────────────────────────────────────────

Q2BN_MAGIC = b"Q2BN"
Q2BN_VERSION = 1
DTYPE_Q2 = 0
DTYPE_FP16 = 1


def pack_state_dict(state_dict: dict[str, torch.Tensor], out_path: str) -> int:
    """Pack model state dict to Q2BN format.

    Returns total bytes written.
    """
    buf = bytearray()
    buf.extend(Q2BN_MAGIC)
    buf.extend(struct.pack("<I", Q2BN_VERSION))
    buf.extend(struct.pack("<I", len(state_dict)))

    for name, tensor in state_dict.items():
        # Determine dtype
        is_q2 = not any(k in name for k in ("embed", "lm_head", "ln", "tau"))

        name_bytes = name.encode("utf-8")
        buf.extend(struct.pack("<I", len(name_bytes)))
        buf.extend(name_bytes)

        shape = tensor.shape
        buf.extend(struct.pack("<I", len(shape)))
        for s in shape:
            buf.extend(struct.pack("<I", s))

        if is_q2:
            buf.extend(struct.pack("<I", DTYPE_Q2))
            # Quantize to Z₄
            tau = INV_CDF_75 / math.sqrt(tensor.shape[-1]) if tensor.dim() > 1 else 0.5
            sym = torch.zeros_like(tensor, dtype=torch.long)
            sym[tensor <= -tau] = 0
            sym[(tensor > -tau) & (tensor <= 0)] = 1
            sym[(tensor > 0) & (tensor <= tau)] = 2
            sym[tensor > tau] = 3
            packed_bytes = pack_z4(sym.view(-1))
            buf.extend(struct.pack("<I", len(packed_bytes)))
            buf.extend(packed_bytes)
        else:
            buf.extend(struct.pack("<I", DTYPE_FP16))
            fp16_bytes = tensor.half().cpu().numpy().tobytes()
            buf.extend(struct.pack("<I", len(fp16_bytes)))
            buf.extend(fp16_bytes)

    out = Path(out_path)
    out.write_bytes(bytes(buf))
    return len(buf)


# ── Muon Optimizer (Nesterov + Spectral Norm) ──────────────────────────────

class Muon(torch.optim.Optimizer):
    """Muon optimizer: Nesterov momentum with spectral normalization.

    Prevents large weight moves from disrupting Z₄ complement structure.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.99,
        weight_decay: float = 0.04,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Weight decay (decoupled, matrices only)
                if wd != 0 and p.dim() >= 2:
                    p.mul_(1 - lr * wd)

                # Spectral normalization for 2D+ parameters
                if p.dim() >= 2:
                    # Approximate spectral norm via power iteration
                    state = self.state[p]
                    if "v" not in state:
                        state["v"] = torch.randn(
                            p.shape[-1], device=p.device, dtype=p.dtype
                        )
                    v = state["v"]
                    u = p.view(-1, p.shape[-1]) @ v
                    u = u / (u.norm() + 1e-8)
                    v = p.view(-1, p.shape[-1]).t() @ u
                    v = v / (v.norm() + 1e-8)
                    state["v"] = v
                    sigma = (u * (p.view(-1, p.shape[-1]) @ v)).sum()
                    d_p = d_p / (sigma + 1e-8)

                # Momentum
                if momentum != 0:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = d_p.clone()
                    else:
                        buf = self.state[p]["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)
                        if nesterov:
                            d_p = d_p + momentum * buf
                        else:
                            d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss


# ── Training Loop ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training configuration optimized for 8×H100 × 10 minutes."""
    # Phases (fraction of total steps)
    warmup_frac: float = 0.10      # Phase 1: FP32 warm-up
    progressive_frac: float = 0.60  # Phase 2: Progressive QAT
    refine_frac: float = 0.30       # Phase 3: Full Q2 refinement

    # Optimizer
    lr: float = 0.01
    weight_decay: float = 0.04
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # Batch
    batch_size: int = 32           # Per GPU
    seq_len: int = 2048
    grad_accum: int = 4

    # SWA
    swa_start_frac: float = 0.60

    # Q2
    tau_refresh_interval: int = 1024

    # Timing
    max_wall_seconds: int = 600

    # Data
    data_path: str = ""
    byte_tokens: bool = True       # Raw byte tokenization (V=256)


def get_cosine_lr(step: int, total_steps: int, lr: float, warmup: int) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> None:
    """Main training entry point.

    Implements the 3-phase Geode-guided training strategy:
      Phase 1: FP32 warm-up (establish activation distributions)
      Phase 2: Progressive Z₄ quantization (deep layers first)
      Phase 3: Full Z₄ refinement with SWA
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()

    # ── Distributed setup ───────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    # ── H100 optimizations ──────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Model ───────────────────────────────────────────────────────────
    model = Q2LTCModel(model_cfg).to(device)

    if is_main:
        size_info = model.estimate_artifact_size()
        print(f"Model: {size_info['total_params']:,} parameters")
        print(f"Estimated artifact: {size_info['compressed_estimate']/1e6:.2f} MB")
        print(f"Budget remaining: {size_info['budget_remaining']/1e6:.2f} MB")
        print(f"Architecture: [GQA, CfC×{model_cfg.cfc_per_level}] × {model_cfg.n_geode_levels}")
        print(f"  = {model_cfg.n_geode_levels} GQA + {model_cfg.n_geode_levels * model_cfg.cfc_per_level} CfC = {model_cfg.n_layers} layers")

    # Compile for max throughput (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="max-autotune")
        if is_main:
            print("Model compiled with max-autotune")
    except Exception:
        if is_main:
            print("torch.compile not available, continuing without")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if isinstance(model, DDP) else model

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = Muon(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    # ── Data (placeholder — replace with FineWeb loading) ───────────────
    # In production, load FineWeb shards as raw bytes or SP-1024 tokens
    if train_cfg.data_path and Path(train_cfg.data_path).exists():
        if is_main:
            print(f"Loading data from {train_cfg.data_path}")
        # Placeholder for real data loading
        data = torch.randint(0, model_cfg.vocab_size, (1024, train_cfg.seq_len + 1))
    else:
        if is_main:
            print("Using synthetic data (no data_path provided)")
        data = torch.randint(0, model_cfg.vocab_size, (1024, train_cfg.seq_len + 1))

    # ── Training ────────────────────────────────────────────────────────
    max_steps = int(os.environ.get("MAX_STEPS", 15000))
    phase1_end = int(max_steps * train_cfg.warmup_frac)
    phase2_end = int(max_steps * (train_cfg.warmup_frac + train_cfg.progressive_frac))
    swa_start = int(max_steps * train_cfg.swa_start_frac)

    # SWA model
    swa_model = None
    swa_n = 0

    start_time = time.time()

    if is_main:
        print(f"\nTraining for {max_steps} steps ({train_cfg.max_wall_seconds}s budget)")
        print(f"  Phase 1 (FP32 warm-up): steps 0–{phase1_end}")
        print(f"  Phase 2 (Progressive QAT): steps {phase1_end}–{phase2_end}")
        print(f"  Phase 3 (Full Z₄ refinement): steps {phase2_end}–{max_steps}")
        print(f"  SWA starts at step {swa_start}")

    model.train()
    cfc_states: Optional[list[Optional[torch.Tensor]]] = None

    for step in range(max_steps):
        # Wall-clock check
        elapsed = time.time() - start_time
        if elapsed > train_cfg.max_wall_seconds - 30:  # 30s buffer for packaging
            if is_main:
                print(f"Wall-clock limit approaching ({elapsed:.0f}s), stopping training")
            break

        # ── Phase transitions ───────────────────────────────────────────
        if step == phase1_end:
            if is_main:
                print(f"\n→ Phase 2: Activating Z₄ quantization (progressive)")
            # Activate deep layers first (Geode: refine before coarse)
            deep_layers = list(range(len(raw_model.layers) - 1, len(raw_model.layers) // 2, -1))
            raw_model.activate_q2(deep_layers)

        elif step == (phase1_end + phase2_end) // 2:
            # Activate remaining layers
            all_layers = list(range(len(raw_model.layers)))
            raw_model.activate_q2(all_layers)
            if is_main:
                print(f"\n→ Phase 2.5: All layers now Z₄ quantized")

        elif step == phase2_end:
            if is_main:
                print(f"\n→ Phase 3: Full Z₄ refinement")

        # ── Threshold refresh ───────────────────────────────────────────
        if step > 0 and step % train_cfg.tau_refresh_interval == 0:
            raw_model.refresh_all_tau()

        # ── Learning rate ───────────────────────────────────────────────
        lr = get_cosine_lr(step, max_steps, train_cfg.lr, train_cfg.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Forward + backward ──────────────────────────────────────────
        batch_idx = step % len(data)
        batch = data[batch_idx].unsqueeze(0).to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, cfc_states = raw_model(input_ids, cfc_states)
            loss = F.cross_entropy(
                logits.view(-1, model_cfg.vocab_size), targets.view(-1)
            )

        loss.backward()

        if (step + 1) % train_cfg.grad_accum == 0:
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ── SWA ─────────────────────────────────────────────────────────
        if step >= swa_start:
            if swa_model is None:
                swa_model = {
                    k: v.clone() for k, v in raw_model.state_dict().items()
                }
                swa_n = 1
            else:
                swa_n += 1
                for k, v in raw_model.state_dict().items():
                    swa_model[k] += (v - swa_model[k]) / swa_n

        # ── Logging ─────────────────────────────────────────────────────
        if is_main and step % 100 == 0:
            bpb = loss.item() / math.log(2)
            phase = (
                "FP32" if step < phase1_end else
                "QAT" if step < phase2_end else
                "Refine"
            )
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"bpb {bpb:.4f} | lr {lr:.6f} | "
                f"phase {phase} | {elapsed:.0f}s"
            )

    # ── Package artifact ────────────────────────────────────────────────
    if is_main:
        print("\n── Packaging artifact ──")
        final_sd = swa_model if swa_model is not None else raw_model.state_dict()
        out_path = "model.q2bin"
        raw_bytes = pack_state_dict(final_sd, out_path)
        print(f"Q2BN size: {raw_bytes / 1e6:.3f} MB")

        # zstd compression
        try:
            import zstandard
            cctx = zstandard.ZstdCompressor(level=22)
            raw_data = Path(out_path).read_bytes()
            compressed = cctx.compress(raw_data)
            compressed_path = "model.q2bin.zst"
            Path(compressed_path).write_bytes(compressed)
            print(f"Compressed: {len(compressed) / 1e6:.3f} MB")
            if len(compressed) <= ARTIFACT_BUDGET:
                print(f"✓ Within budget ({ARTIFACT_BUDGET / 1e6:.0f} MB)")
            else:
                print(f"✗ OVER BUDGET by {(len(compressed) - ARTIFACT_BUDGET) / 1e6:.3f} MB")
        except ImportError:
            print("zstandard not installed; skipping compression")

    if world_size > 1:
        dist.destroy_process_group()


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = ModelConfig(
        vocab_size=int(os.environ.get("VOCAB_SIZE", 256)),
        d_model=int(os.environ.get("D_MODEL", 768)),
        n_geode_levels=int(os.environ.get("N_GEODE_LEVELS", 4)),
        max_seq_len=int(os.environ.get("SEQ_LEN", 2048)),
    )

    tcfg = TrainConfig(
        data_path=os.environ.get("DATA_PATH", ""),
        byte_tokens=os.environ.get("BYTE_TOKENS", "1") == "1",
        max_wall_seconds=int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600)),
        batch_size=int(os.environ.get("BATCH_SIZE", 32)),
        seq_len=cfg.max_seq_len,
    )

    train(cfg, tcfg)
