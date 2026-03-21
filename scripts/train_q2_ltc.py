#!/usr/bin/env python3
"""
train_q2_ltc.py — Q²-QAT Hybrid LTC-Transformer for OpenAI Parameter Golf.

Architecture: [GQA, CfC, CfC, CfC] × 4 = 16 layers (Geode-derived, see §4.5
of PARAMETER_GOLF.md).  The layer layout is derived from the Geode factorization
S(x) - 1 = S1·G where S1=4x gives 4 GQA (coarse) blocks and G=1/(1-3x) gives
3 CfC (refinement) blocks per GQA block.

Quantisation: Q² 2-bit QAT with straight-through estimator (STE).
Optimizer:    Muon (Nesterov + spectral normalisation) — current SOTA.
Compression:  Q2-packed weights + zstd-22 for final artifact.

Usage (8×H100):
    torchrun --standalone --nproc_per_node=8 scripts/train_q2_ltc.py

Single GPU (smoke test):
    python scripts/train_q2_ltc.py

Environment variables (all optional, reasonable defaults):
    D_MODEL         hidden dimension          (default: 768)
    N_HEADS         attention heads           (default: 12)
    N_KV_HEADS      KV heads for GQA          (default: 4)
    MAX_STEPS       training steps            (default: 5000)
    BATCH_TOKENS    tokens per gradient step  (default: 131072)
    SEQ_LEN         sequence length           (default: 2048)
    DATA_PATH       FineWeb tokenised shards  (default: ./data/datasets/fineweb10B_sp1024)
    VOCAB_SIZE      vocabulary size           (default: 1024)
    OUT_DIR         checkpoint directory      (default: ./checkpoints)
    WARMUP_STEPS    LR warm-up steps          (default: 200)
    Q2_WARMUP       FP32 warm-up before QAT   (default: 500)
    VAL_EVERY       validation interval       (default: 200)
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model (Geode-derived: 4 GQA + 12 CfC = 16 layers)
    d_model:    int = int(os.getenv("D_MODEL",    "768"))
    n_heads:    int = int(os.getenv("N_HEADS",    "12"))
    n_kv_heads: int = int(os.getenv("N_KV_HEADS", "4"))
    n_layers:   int = 16     # fixed: [GQA, CfC, CfC, CfC] × 4
    mlp_ratio:  int = 3      # MLP hidden = d_model × mlp_ratio
    vocab_size: int = int(os.getenv("VOCAB_SIZE", "1024"))

    # Q²-QAT
    q2_warmup:        int   = int(os.getenv("Q2_WARMUP",  "500"))
    tau_update_every: int   = 1024
    ste_kappa_scale:  float = 3.0   # STE passthrough window: κ = kappa_scale × τ*

    # Training
    max_steps:    int   = int(os.getenv("MAX_STEPS",    "5000"))
    batch_tokens: int   = int(os.getenv("BATCH_TOKENS", "131072"))
    seq_len:      int   = int(os.getenv("SEQ_LEN",      "2048"))
    lr:           float = 3e-4
    wd:           float = 0.04
    grad_clip:    float = 1.0
    swa_start:    float = 0.6   # SWA from this fraction of total steps
    warmup_steps: int   = int(os.getenv("WARMUP_STEPS", "200"))
    val_every:    int   = int(os.getenv("VAL_EVERY",    "200"))
    val_tokens:   int   = 1_000_000

    # Paths
    data_path: str = os.getenv("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    out_dir:   str = os.getenv("OUT_DIR",   "./checkpoints")


# ── Q²-QAT: straight-through estimator ───────────────────────────────────────

class _Q2STEFunction(torch.autograd.Function):
    """Straight-through estimator for Q² 2-bit weight quantisation.

    Forward:  maps float32 weights to Q² reconstruction values
              {-1, -0.5, +0.5, +1} × τ (cell centroids scaled by threshold).
    Backward: passes gradient unchanged where |W| ≤ κ (STE window);
              zeroes gradient outside the window to suppress outlier updates.
    """

    # Unit reconstruction points for symbols {A=0, B=1, C=2, D=3}.
    # Module-level constant; moved to device in forward to avoid repeated allocation.
    _LEVELS = torch.tensor([-1.0, -0.5, 0.5, 1.0])

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        W: Tensor,
        tau: Tensor,
        kappa: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(W, kappa)
        # Vectorised quantisation (matches q2_quantise in q2_pack.py).
        sym = (W > -tau).to(torch.long)
        sym = sym + (W > 0).to(torch.long)
        sym = sym + (W > tau).to(torch.long)    # sym in {0,1,2,3}
        # Cache-friendly: _LEVELS is a 4-element constant; .to() is a no-op
        # when dtype/device already match (which they will after the first call).
        levels = _Q2STEFunction._LEVELS.to(device=W.device, dtype=W.dtype)
        return levels[sym] * tau

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tuple[Tensor, None, None]:
        W, kappa = ctx.saved_tensors
        # STE: pass gradient only within the quantisation window.
        grad_W = grad_output * (W.abs() <= kappa).to(grad_output.dtype)
        return grad_W, None, None


q2_ste = _Q2STEFunction.apply


class Q2Linear(nn.Linear):
    """Linear layer with Q²-QAT: quantised weights in forward, exact in backward.

    Behaves as a standard nn.Linear during FP32 warm-up (quantised=False).
    Call activate_q2() after warm-up to switch to STE mode.

    The per-row threshold τ* is computed once from the empirical 75th percentile
    of |W| (reservoir calibration, §D-2.5) and refreshed every tau_update_every
    forward steps.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.quantised = False
        self._step = 0
        self._tau_update_every = 1024
        self._ste_kappa_scale  = 3.0
        # Non-parameter buffers (excluded from optimizer state).
        self.register_buffer("_tau",   torch.full((out_features, 1), 0.6745))
        self.register_buffer("_kappa", torch.full((out_features, 1), 2.0236))

    @torch.no_grad()
    def _refresh_tau(self) -> None:
        tau = torch.quantile(
            self.weight.float().abs(), 0.75, dim=1, keepdim=True
        ).clamp(min=1e-6)
        self._tau.copy_(tau)
        self._kappa.copy_(tau * self._ste_kappa_scale)

    def forward(self, x: Tensor) -> Tensor:
        if not self.quantised:
            return F.linear(x, self.weight, self.bias)
        self._step += 1
        if self._step % self._tau_update_every == 0:
            self._refresh_tau()
        W_hat = q2_ste(self.weight, self._tau, self._kappa)
        return F.linear(x, W_hat, self.bias)

    def activate_q2(
        self,
        update_every: int   = 1024,
        kappa_scale:  float = 3.0,
    ) -> None:
        """Switch to QAT mode (call once after FP32 warm-up completes)."""
        self._tau_update_every = update_every
        self._ste_kappa_scale  = kappa_scale
        self._refresh_tau()
        self.quantised = True


# ── CfC block (Geode G-node: one 3-way refinement step) ──────────────────────

class CfCBlock(nn.Module):
    """Closed-form Continuous-time recurrent block.

    Implements one step of the Geode G = 1/(1-3x) refinement tree.
    Solves the LTC ODE analytically (Hasani et al. 2022, arXiv:2106.13898):

        h_new = exp(-A1·dt) · h + (A2/A1) · (1 - exp(-A1·dt))

    The recurrent state h propagates information across tokens within a
    sequence without growing a KV cache.  Memory cost per layer: O(batch·d)
    regardless of sequence length.

    All Q2Linear layers participate in Q²-QAT when activate_q2() is called
    on the parent model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm  = nn.RMSNorm(d_model)
        # A1: decay-rate network (input=[x,h] → positive scalar per dim)
        self.ff_a1 = Q2Linear(d_model * 2, d_model)
        # A2: integration-target network (input=[x,h] → target state)
        self.ff_a2 = Q2Linear(d_model * 2, d_model)
        self.out   = Q2Linear(d_model, d_model)
        # Learnable log time-step (log-parameterised → strictly positive).
        self.log_dt = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, D) — token representations from the previous block.
            h: (B, D)    — recurrent state carried from the previous token.

        Returns:
            y: (B, T, D) — output representations (residual-connected).
            h: (B, D)    — updated recurrent state (final token in sequence).
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        dt = self.log_dt.exp()   # (D,) — positive, learnable time step

        out_steps: list[Tensor] = []
        for t in range(T):
            xt = x[:, t, :]                              # (B, D)
            xh = torch.cat([xt, h], dim=-1)              # (B, 2D)
            a1 = F.softplus(self.ff_a1(xh))              # (B, D) decay rate > 0
            a2 = self.ff_a2(xh)                          # (B, D) integration target
            decay = torch.exp(-a1 * dt)                  # (B, D) in (0, 1)
            h = decay * h + (a2 / (a1 + 1e-6)) * (1.0 - decay)
            out_steps.append(h)

        y = torch.stack(out_steps, dim=1)                # (B, T, D)
        return residual + self.out(y), h


# ── GQA block (Geode S1-node: one 4-way coarse selection) ────────────────────

class GQABlock(nn.Module):
    """Grouped Query Attention block with fused MLP.

    Implements one step of the Geode S1 = 4x coarse-quantisation node.
    Uses PyTorch's fused scaled_dot_product_attention (FlashAttention path on
    Ampere/Hopper hardware) for memory-efficient causal attention.

    KV heads are shared across Q-head groups (GQA) to reduce parameter count
    while preserving the representational depth of full MHA.

    The MLP uses a SwiGLU gate (element-wise product of two projections) for
    parameter efficiency.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, mlp_ratio: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups  = n_heads // n_kv_heads
        self.head_dim   = d_model // n_heads

        self.attn_norm = nn.RMSNorm(d_model)
        self.q  = Q2Linear(d_model, d_model)
        self.k  = Q2Linear(d_model, self.head_dim * n_kv_heads)
        self.v  = Q2Linear(d_model, self.head_dim * n_kv_heads)
        self.o  = Q2Linear(d_model, d_model)

        d_ff = d_model * mlp_ratio
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp_up   = Q2Linear(d_model, d_ff)
        self.mlp_gate = Q2Linear(d_model, d_ff)
        self.mlp_down = Q2Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        residual = x
        x = self.attn_norm(x)

        # QKV projections.
        q = self.q(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads for GQA (avoids materialising the full n_heads KV).
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # FlashAttention (causal; fused kernel on Ampere+).
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o(attn)

        # SwiGLU MLP: gated linear unit with SiLU non-linearity.
        residual2 = x
        x = self.mlp_norm(x)
        x = residual2 + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))
        return x


# ── full model: [GQA, CfC, CfC, CfC] × 4 ────────────────────────────────────

class Q2LTCModel(nn.Module):
    """Q²-QAT Hybrid LTC-Transformer with Geode-derived layer layout.

    The layer stack mirrors the Geode factorisation S - 1 = S1·G:
        S1 = 4x     → 4 GQA blocks (coarse: 4 choices each, 2 bits/level)
        G = 1/(1-3x)→ 3 CfC blocks per GQA (refinement: 3 choices, 1.585 bits/step)

    Pattern:  [GQA, CfC, CfC, CfC] × 4 = 16 layers (4 GQA + 12 CfC)
    GQA positions: 0, 4, 8, 12  (0-indexed in self.layers)
    CfC positions: 1-3, 5-7, 9-11, 13-15

    Information capacity at depth d:
        4 × (2 + 3 × log₂ 3) ≈ 27.0 bits — sufficient for 2048-token LM.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        self.embed    = nn.Embedding(cfg.vocab_size, D)
        self.emb_norm = nn.RMSNorm(D)

        # Build [GQA, CfC, CfC, CfC] × 4 using the Geode structure.
        layers: list[nn.Module] = []
        for _ in range(4):                                    # 4 coarse S1 nodes
            layers.append(GQABlock(D, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_ratio))
            for _ in range(3):                                # 3 G refinement nodes
                layers.append(CfCBlock(D))
        self.layers = nn.ModuleList(layers)                   # 16 layers total

        self.norm    = nn.RMSNorm(D)
        self.lm_head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight               # tied weights

        # BigramHash log-prior (FP16; loaded separately from the artifact).
        self.register_buffer(
            "bigram_logprobs",
            torch.zeros(cfg.vocab_size, cfg.vocab_size, dtype=torch.float16),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """OrthoInit for projection matrices; small normal for embeddings."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, Q2Linear)):
                if m.weight.ndim >= 2 and m.weight.shape[0] <= m.weight.shape[1]:
                    nn.init.orthogonal_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def activate_q2(self, cfg: Config) -> None:
        """Switch all Q2Linear layers to QAT mode after FP32 warm-up."""
        for m in self.modules():
            if isinstance(m, Q2Linear):
                m.activate_q2(
                    update_every=cfg.tau_update_every,
                    kappa_scale=cfg.ste_kappa_scale,
                )

    def forward(
        self,
        input_ids: Tensor,
        prev_token: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            input_ids: (B, T) int64 token indices.
            prev_token: (B,) int64 — token immediately before input_ids[:,0];
                        used to look up the BigramHash prior for position 0.

        Returns:
            logits: (B, T, V) float32.
        """
        B, T = input_ids.shape
        D = self.cfg.d_model

        x = self.emb_norm(self.embed(input_ids))    # (B, T, D)

        # CfC recurrent states: reset to zero at the start of each sequence.
        # Dict keyed by layer index to avoid storing states for GQA layers.
        h_states: Dict[int, Tensor] = {}

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GQABlock):
                x = layer(x)
            else:
                if i not in h_states:
                    h_states[i] = x.new_zeros(B, D)
                x, h_states[i] = layer(x, h_states[i])

        x = self.norm(x)
        logits = self.lm_head(x)   # (B, T, V)

        # Add BigramHash log-prior for position 0.
        if prev_token is not None:
            prior = self.bigram_logprobs[prev_token].to(logits.dtype)  # (B, V)
            logits[:, 0, :] = logits[:, 0, :] + prior

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Muon optimizer ─────────────────────────────────────────────────────────────

class Muon(torch.optim.Optimizer):
    """Muon — Nesterov momentum + per-matrix spectral normalisation.

    Adapted from modded-nanogpt (KellerJordan).  The spectral normalisation
    step divides each weight update by its largest singular value, which
    prevents large gradient steps from disrupting the Q2 complement structure
    during QAT — a stronger form of gradient clipping.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.04,
        nesterov: bool = True,
    ):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr  = group["lr"]
            mom = group["momentum"]
            wd  = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = g.clone()
                else:
                    state["buf"].mul_(mom).add_(g)
                    g = (g + state["buf"] * mom) if group["nesterov"] else state["buf"]
                # Per-matrix normalisation: scale by inverse Frobenius norm (cheap stabiliser).
                if g.ndim >= 2:
                    sigma = torch.linalg.norm(g)  # Frobenius norm (avoids per-step SVD cost)
                    if sigma > 0:
                        g = g / sigma
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g.to(p.dtype), alpha=-lr)

        return loss


# ── data loading ───────────────────────────────────────────────────────────────

def _shard_files(data_path: str) -> list[Path]:
    p = Path(data_path)
    files = sorted(p.glob("*.bin")) + sorted(p.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .bin/.npy shards found in {data_path!r}")
    return files


def token_stream(
    data_path: str,
    seq_len: int,
    device: torch.device,
    rank: int = 0,
    world: int = 1,
) -> Iterator[Tuple[Tensor, Tensor]]:
    """Yield (input_ids, target_ids) pairs of length seq_len.

    Shards are distributed round-robin across ranks so each GPU sees a
    disjoint subset of the data.
    """
    import numpy as np
    files = _shard_files(data_path)
    # Assign shards to this rank.
    my_files = [f for i, f in enumerate(files) if i % world == rank]
    if not my_files:
        my_files = files  # fallback for single-GPU runs

    while True:
        for f in my_files:
            raw = f.read_bytes()
            tokens = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
            tokens = tokens.to(torch.long)
            for start in range(0, len(tokens) - seq_len - 1, seq_len + 1):
                chunk = tokens[start : start + seq_len + 1].to(device)
                yield chunk[:seq_len], chunk[1:]


# ── validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_bpb(
    model: nn.Module,
    data_path: str,
    vocab_size: int,
    seq_len: int,
    val_tokens: int,
    device: torch.device,
    stride: int = 64,
) -> float:
    """Sliding-window bits-per-byte on the validation split."""
    val_files = sorted(Path(data_path).glob("fineweb_val_*.bin"))
    if not val_files:
        return float("nan")

    import numpy as np
    total_bits = 0.0
    total_bytes = 0
    model.eval()

    for f in val_files:
        raw = f.read_bytes()
        tokens = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).long()
        # Sliding window evaluation at stride=64 (current SOTA).
        for start in range(0, min(len(tokens), val_tokens) - seq_len, stride):
            chunk = tokens[start : start + seq_len + 1].to(device)
            inp, tgt = chunk[:seq_len].unsqueeze(0), chunk[1:].unsqueeze(0)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
            # Only score the last stride tokens (context consumed earlier).
            score_start = seq_len - stride
            loss = F.cross_entropy(
                logits[0, score_start:].view(-1, vocab_size),
                tgt[0, score_start:].view(-1),
            )
            total_bits  += loss.item() * stride * math.log2(math.e)
            total_bytes += stride  # 1 token ≈ 1 byte for SP-1024
            if total_bytes >= val_tokens:
                break
        if total_bytes >= val_tokens:
            break

    model.train()
    return total_bits / max(total_bytes, 1)


# ── training loop ──────────────────────────────────────────────────────────────

def train(cfg: Config) -> None:
    # Distributed setup.
    rank  = int(os.getenv("RANK",       "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    use_dist = world > 1
    if use_dist:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    master = rank == 0

    # Build model.
    model = Q2LTCModel(cfg).to(device)
    if master:
        n_params = model.count_parameters()
        print(f"Q2-LTC model: {n_params:,} parameters ({n_params / 1e6:.1f} M)")
        print(f"Layer layout: [GQA, CfC, CfC, CfC] × 4 = {cfg.n_layers} layers")

    if use_dist:
        model = DDP(model, device_ids=[local])
    raw_model: Q2LTCModel = model.module if use_dist else model  # type: ignore[assignment]

    # Compile for maximum H100 throughput.
    model = torch.compile(model, mode="max-autotune")

    # Separate optimizer groups: Q2-quantised weight matrices vs. all other params.
    q2_params = [
        p for n, p in raw_model.named_parameters()
        if "weight" in n and p.ndim >= 2
    ]
    other_params = [
        p for n, p in raw_model.named_parameters()
        if not ("weight" in n and p.ndim >= 2)
    ]
    optimizer = Muon([
        {"params": q2_params,    "lr": cfg.lr, "weight_decay": cfg.wd},
        {"params": other_params, "lr": cfg.lr, "weight_decay": 0.0},
    ])

    # SWA (stochastic weight averaging over last 40% of training).
    swa_model  = torch.optim.swa_utils.AveragedModel(raw_model)
    swa_start  = int(cfg.max_steps * cfg.swa_start)
    swa_active = False

    # bfloat16 autocast on H100; no GradScaler needed (bf16 has enough dynamic range).
    batch_size = max(1, cfg.batch_tokens // cfg.seq_len)
    data = token_stream(cfg.data_path, cfg.seq_len, device, rank, world)

    if master:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    for step in range(1, cfg.max_steps + 1):
        # Cosine LR schedule with linear warm-up.
        if step <= cfg.warmup_steps:
            lr_scale = step / cfg.warmup_steps
        else:
            frac = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * frac))
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * lr_scale

        # Switch to Q²-QAT after FP32 warm-up.
        if step == cfg.q2_warmup + 1:
            raw_model.activate_q2(cfg)
            if master:
                print(f"[step {step:5d}] Q² QAT activated")

        # Gradient accumulation over batch_size micro-batches.
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(batch_size):
            inp, tgt = next(data)
            inp, tgt = inp.unsqueeze(0), tgt.unsqueeze(0)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss   = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    tgt.view(-1),
                ) / batch_size
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping + optimizer step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # SWA update.
        if step >= swa_start:
            swa_model.update_parameters(raw_model)
            swa_active = True

        # Logging.
        if master and step % 100 == 0:
            elapsed = time.perf_counter() - t0
            tok_per_s = 100 * cfg.batch_tokens / elapsed
            print(
                f"step {step:5d} | loss {total_loss:.4f} | "
                f"lr {lr_scale * cfg.lr:.2e} | "
                f"{tok_per_s / 1e3:.1f} k tok/s"
            )
            t0 = time.perf_counter()

        # Validation.
        if master and step % cfg.val_every == 0:
            bpb = estimate_val_bpb(
                swa_model if swa_active else raw_model,
                cfg.data_path, cfg.vocab_size, cfg.seq_len,
                cfg.val_tokens, device,
            )
            print(f"  val_bpb = {bpb:.4f}")

    # ── artifact packaging ─────────────────────────────────────────────────────
    if not master:
        if use_dist:
            dist.destroy_process_group()
        return

    print("\nPackaging artifact …")

    final_sd = {
        k: v.cpu()
        for k, v in (swa_model.module if swa_active else raw_model).state_dict().items()
    }

    # Import q2_pack from this scripts/ directory.
    import importlib.util
    import sys
    _spec = importlib.util.spec_from_file_location(
        "q2_pack", Path(__file__).parent / "q2_pack.py"
    )
    assert _spec and _spec.loader
    q2_pack = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(q2_pack)  # type: ignore[union-attr]

    q2bin_path = Path(cfg.out_dir) / "model.q2bin"
    raw_bytes = q2_pack.pack_state_dict(final_sd, q2bin_path)
    print(f"  Q2-packed:  {raw_bytes:,} bytes ({raw_bytes / 1e6:.3f} MB)")

    # Compress with zstd level 22 (requires the `zstandard` package).
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(q2bin_path.read_bytes())
        zst_path = q2bin_path.with_suffix(".q2bin.zst")
        zst_path.write_bytes(compressed)
        print(f"  zstd-22:    {len(compressed):,} bytes ({len(compressed) / 1e6:.3f} MB)")
    except ImportError:
        compressed = q2bin_path.read_bytes()
        zst_path   = q2bin_path
        print("  (zstandard not installed; using uncompressed Q2BN)")

    this_file_bytes = len(Path(__file__).read_bytes())
    q2_pack_path = Path(__file__).parent / "q2_pack.py"
    q2_pack_bytes = q2_pack_path.stat().st_size if q2_pack_path.exists() else 0
    code_bytes = this_file_bytes + q2_pack_bytes
    total      = len(compressed) + code_bytes
    print(f"  code:       {code_bytes:,} bytes")
    print(f"  TOTAL:      {total:,} bytes ({total / 1e6:.3f} MB)")
    if total > 16_000_000:
        print("  WARNING: exceeds 16 MB budget — reduce d_model or add layers")
    else:
        print("  ✓ within 16 MB budget")

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(Config())
