# Parameter Golf: Revised Strategy (PyTorch-Native Q²)

> **Status**: Revised based on feedback
> **Supersedes**: PARAMETER_GOLF_APPROACH.md (initial strategy)
> **Related**: [DESIGN.md](DESIGN.md), [RELATED_WORK.md](RELATED_WORK.md)

## Executive Summary

Based on critical feedback, this revised strategy focuses on:

1. **Pure PyTorch/GPU implementation** - No WASM, leverage 8×H100 GPUs directly
2. **Even bit widths (power-of-2 where possible)** - Z₄ (2-bit), Z₈ (4-bit), Z₁₂ (6-bit transitional), Z₁₆ (8-bit)
3. **Cache-line optimization** - Design for 64-byte cache lines and SIMD registers
4. **Geode-guided architecture** - Exploit hierarchical structure for efficiency

**Target**: Sub-1.10 bits/byte on FineWeb validation, <16MB artifact, <10 min training on 8×H100

---

## 1. Why the Original Approach Needed Revision

### 1.1 WASM Misconception

**Original error**: Proposed extending `src/q2.wat` for weight quantization and using WASM for training.

**Correction**: Parameter Golf provides 8×H100 GPUs for training. WASM:
- Cannot leverage GPUs effectively
- Adds unnecessary complexity for training
- Was designed for browser inference, not GPU training

**New direction**: Pure PyTorch implementation optimized for H100 tensor cores.

### 1.2 int5 Instability

**Original error**: Proposed int5 quantization (5 bits per weight) following leaderboard trends.

**Correction**: Odd bit widths are mathematically problematic:
- p-adic numbers unstable at odd values
- Cache line alignment requires power-of-2 widths
- Cannot efficiently pack into 64-byte cache lines

**Example**: int5 packing is messy:
```
5 bits × 12.8 = 64 bits (not integer number of weights per register)
```

vs. power-of-2:
```
2 bits × 32 = 64 bits (Z₄: 32 weights per register)
4 bits × 16 = 64 bits (Z₈: 16 weights per register)
6 bits × 10 + 4 padding = 64 bits (Z₁₂: requires padding)
8 bits × 8 = 64 bits (Z₁₆: 8 weights per register, perfect alignment)
```

### 1.3 The Z_N Ring Hierarchy

From feedback: Consider Z_N rings as natural extensions of Z₄.

**Key insight**: Nature runs on Z₄ (DNA base pairs), but codons (triplets) and higher structures suggest hierarchical composition.

The Geode factorization S - 1 = S₁ · G suggests:
- **S₁**: Coarse quantization at lower precision (Z₄, 2-bit)
- **G**: Refinement at higher precision (Z₈, Z₁₆)

This is not arbitrary mixing—it's the mathematical structure of hierarchical quantization.

---

## 2. Revised Architecture

### 2.1 Quantization Hierarchy

Instead of uniform int5/int6, use **structured mixed precision** guided by Geode:

| Component | Ring | Bits | Rationale |
|-----------|------|------|-----------|
| **Embedding** | Z₁₆ | 8 | High-capacity input/output layer |
| **Early layers (1-4)** | Z₈ | 4 | Coarse feature extraction |
| **Middle layers (5-8)** | Z₁₂ | 6 | Transition zone (compromise) |
| **Deep layers (9-12)** | Z₄ | 2 | Structural encoding (minimal) |

**Mathematical justification**:
- Z₄ ⊂ Z₈ ⊂ Z₁₆ (2 | 4 | 8, natural divisibility)
- Z₁₂ = 3 × Z₄ (codon-like structure)
- Progressive refinement matches Geode factorization

### 2.2 Model Architecture

```python
# Simplified architecture (no LTC complexity initially)

Input (BigramHash 10240 vocab)
  ↓
Embedding (384 dim, Z₁₆/8-bit) [3.9M params, 3.9MB]
  ↓
4× Attention blocks (Z₈/4-bit) [~4M params, 2MB]
  ↓
4× Attention blocks (Z₁₂/6-bit) [~4M params, 3MB]
  ↓
4× Attention blocks (Z₄/2-bit) [~4M params, 1MB]
  ↓
Output (tied, Z₁₆/8-bit) [0 params]
```

**Total**: ~12M params, ~10MB compressed (37% headroom)

**Note**: Start with standard attention, not LTC. Proven architecture first, then optimize.

### 2.3 Cache-Line Optimized Packing

Design for 64-byte (512-bit) cache lines on H100:

```python
# Z₄ (2-bit): Perfect alignment
# 32 weights × 2 bits = 64 bits = 1 register
# 256 weights = 8 registers = 64 bytes = 1 cache line

class Z4Quantizer:
    @staticmethod
    def pack_cache_line(weights: torch.Tensor) -> torch.Tensor:
        """Pack 256 weights into 64 bytes (1 cache line)"""
        assert weights.shape[-1] % 256 == 0
        # Quantize to {0, 1, 2, 3}
        quantized = quantize_z4(weights)  # → [batch, ..., n]
        # Pack 4 symbols per byte (Gray encoded)
        packed = pack_4_per_byte(quantized)  # → [batch, ..., n//4]
        return packed

# Z₈ (4-bit): Perfect alignment
# 16 weights × 4 bits = 64 bits = 1 register
# 128 weights = 8 registers = 64 bytes = 1 cache line

class Z8Quantizer:
    @staticmethod
    def pack_cache_line(weights: torch.Tensor) -> torch.Tensor:
        """Pack 128 weights into 64 bytes (1 cache line)"""
        assert weights.shape[-1] % 128 == 0
        quantized = quantize_z8(weights)  # → {0..7}
        packed = pack_2_per_byte(quantized)  # → [batch, ..., n//2]
        return packed

# Z₁₆ (8-bit): Perfect alignment
# 8 weights × 8 bits = 64 bits = 1 register
# 64 weights = 8 registers = 64 bytes = 1 cache line

class Z16Quantizer:
    @staticmethod
    def pack_cache_line(weights: torch.Tensor) -> torch.Tensor:
        """Pack 64 weights into 64 bytes (1 cache line)"""
        assert weights.shape[-1] % 64 == 0
        quantized = quantize_z16(weights)  # → {0..15}
        return quantized.to(torch.uint8)  # 1 byte per weight
```

---

## 3. PyTorch Implementation Strategy

### 3.1 Core Quantization Kernel

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Q2Quantize(torch.autograd.Function):
    """
    Q² quantization to Z₄ (2-bit) with Gray encoding
    Optimized for H100 tensor cores
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Quantize weights to {0, 1, 2, 3} = {A, B, C, D}

        Args:
            weight: Input weights (any shape)
            tau: Equiprobable threshold

        Returns:
            Quantized weights in range [0, 3]
        """
        # Vectorized quantization (GPU-friendly)
        sym = torch.zeros_like(weight, dtype=torch.long)
        sym = torch.where(weight <= -tau, 0, sym)  # A: strong negative
        sym = torch.where((weight > -tau) & (weight <= 0), 1, sym)  # B: weak negative
        sym = torch.where((weight > 0) & (weight <= tau), 2, sym)  # C: weak positive
        sym = torch.where(weight > tau, 3, sym)  # D: strong positive

        # Gray encode: g = sym ⊕ (sym >> 1)
        gray = sym ^ (sym >> 1)

        # Dequantize for forward pass
        # Map {0, 1, 2, 3} → {-1.5τ, -0.5τ, +0.5τ, +1.5τ}
        dequant_map = torch.tensor(
            [-1.5, -0.5, 0.5, 1.5],
            dtype=weight.dtype,
            device=weight.device
        )
        weight_q = dequant_map[gray] * tau

        ctx.save_for_backward(weight, weight_q, torch.tensor(tau))
        return weight_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Straight-through estimator (STE)"""
        return grad_output, None


class Q2Linear(nn.Module):
    """
    Linear layer with Q² quantization
    Supports Z₄, Z₈, Z₁₂, Z₁₆
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        z_ring: int = 4,  # 4, 8, 12, or 16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.z_ring = z_ring
        self.bits = {4: 2, 8: 4, 12: 6, 16: 8}[z_ring]

        # Full-precision weights (will be quantized during forward)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Compute equiprobable threshold
        self.register_buffer(
            'tau',
            torch.tensor(0.6745 / (in_features ** 0.5))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights during training
        if self.training:
            weight_q = Q2Quantize.apply(self.weight, self.tau)
        else:
            # Use cached quantized weights during inference
            weight_q = self.weight_quantized if hasattr(self, 'weight_quantized') else self.weight

        return F.linear(x, weight_q, self.bias)

    def finalize_quantization(self):
        """Call before exporting model"""
        with torch.no_grad():
            self.weight_quantized = Q2Quantize.apply(self.weight, self.tau)
            # Can delete full-precision weights to save memory
            del self.weight
```

### 3.2 Geode-Guided Progressive Training

```python
def train_progressive(
    model: nn.Module,
    dataloader: DataLoader,
    max_steps: int = 15000,
):
    """
    Three-phase Geode-guided training:
    Phase 1: All layers at Z₁₆ (8-bit) - learn coarse structure
    Phase 2: Progressive quantization layer-by-layer
    Phase 3: Fine-tune at target precision
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Phase 1: Coarse learning (30% of steps)
    phase1_steps = int(max_steps * 0.3)
    for step in range(phase1_steps):
        loss = train_step(model, dataloader, optimizer)
        if step % 100 == 0:
            print(f"Phase 1 [{step}/{phase1_steps}]: loss={loss:.4f}")

    # Phase 2: Progressive quantization (50% of steps)
    phase2_steps = int(max_steps * 0.5)
    layers_to_quantize = get_quantizable_layers(model)

    for layer_idx, layer in enumerate(layers_to_quantize):
        # Quantize this layer
        quantize_layer(layer, target_z_ring=get_target_z_ring(layer_idx))

        # Fine-tune for a few steps
        for step in range(phase2_steps // len(layers_to_quantize)):
            loss = train_step(model, dataloader, optimizer)

        print(f"Quantized layer {layer_idx}, loss={loss:.4f}")

    # Phase 3: Final fine-tuning (20% of steps)
    phase3_steps = max_steps - phase1_steps - phase2_steps
    for step in range(phase3_steps):
        loss = train_step(model, dataloader, optimizer)
        if step % 100 == 0:
            print(f"Phase 3 [{step}/{phase3_steps}]: loss={loss:.4f}")


def get_target_z_ring(layer_idx: int) -> int:
    """Assign Z-ring based on layer depth (Geode hierarchy)"""
    if layer_idx < 4:
        return 8  # Z₈ (4-bit) for early layers
    elif layer_idx < 8:
        return 12  # Z₁₂ (6-bit) for middle layers
    else:
        return 4  # Z₄ (2-bit) for deep layers
```

### 3.3 H100 Optimization

```python
# Use bfloat16 for non-quantized operations (H100 native)
torch.set_float32_matmul_precision('high')

# Enable TF32 for matmul (H100 feature)
torch.backends.cuda.matmul.allow_tf32 = True

# Compile model for faster execution (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')

# Use CUDA graphs for fixed-size batches
# (eliminates kernel launch overhead)
```

---

## 4. Why This Beats int5 Approaches

### 4.1 Cache-Line Efficiency

**int5 (current SOTA)**:
```
5 bits/weight → 12.8 weights per 64-bit register
Cannot align to cache lines without padding
Memory access patterns irregular
```

**Q² Z₄ (our approach)**:
```
2 bits/weight → 32 weights per 64-bit register
Perfect cache line alignment (256 weights = 64 bytes)
SIMD-friendly (AVX-512 processes 32 weights at once)
```

**Speedup estimate**: 1.3-1.5× faster matmul due to better memory access patterns.

### 4.2 Mathematical Stability

From feedback: p-adic numbers unstable at odd values.

**int5**: No ring structure, arbitrary quantization levels
**Z₄, Z₈, Z₁₆**: Proper rings with well-defined arithmetic

Example: Adding two Z₄ values:
```python
# Z₄ arithmetic (modulo 4)
a = 2  # C (weak positive)
b = 3  # D (strong positive)
c = (a + b) % 4 = 1  # B (weak negative after wrap)

# This preserves cyclic structure
# Lee distance: d_L(c, a) = min(|1-2|, 4-|1-2|) = 1
```

int5 has no such structure—quantization levels are ad-hoc.

### 4.3 Geode Hierarchy

The progression Z₄ → Z₈ → Z₁₆ matches natural hierarchy:

- **DNA**: 4 bases (Z₄)
- **Codons**: 64 = 4³ combinations
- **Amino acids**: 20 encoded by codons

Our architecture mirrors this:
- **Deep layers**: Z₄ (structural encoding, like DNA)
- **Middle layers**: Z₁₂ = 3×Z₄ (codon-like composition)
- **Shallow layers**: Z₁₆ (high-capacity, like protein structures)

---

## 5. Revised Parameter Budget

### 5.1 Target Configuration

```python
vocab_size = 10240  # BigramHash
d_model = 512       # Wider than original (better for quantization)
n_layers = 12
n_heads = 8
mlp_ratio = 4       # Standard 4× expansion

# Parameter counts by component:
embedding = vocab_size × d_model = 5.24M params
layer_1_4 (Z₈) = 4 × (4×d_model² + d_model×mlp_ratio) ≈ 4.2M params
layer_5_8 (Z₁₂) = 4 × (4×d_model² + d_model×mlp_ratio) ≈ 4.2M params
layer_9_12 (Z₄) = 4 × (4×d_model² + d_model×mlp_ratio) ≈ 4.2M params
output (tied) = 0 params

Total = 17.84M params
```

### 5.2 Compressed Size

```python
# Embedding (Z₁₆, 8-bit)
embedding_size = 5.24M × 1 byte = 5.24 MB

# Layers 1-4 (Z₈, 4-bit)
layer_1_4_size = 4.2M × 0.5 bytes = 2.1 MB

# Layers 5-8 (Z₁₂, 6-bit)
layer_5_8_size = 4.2M × 0.75 bytes = 3.15 MB

# Layers 9-12 (Z₄, 2-bit)
layer_9_12_size = 4.2M × 0.25 bytes = 1.05 MB

# Total before compression
subtotal = 11.54 MB

# After zstd-22 compression (typical 0.8×)
final_size ≈ 9.2 MB

# Headroom: 16 MB - 9.2 MB = 6.8 MB (42%)
```

**Conclusion**: Significant headroom allows for:
- Larger model (18M → 22M params)
- Higher precision for critical layers
- Additional optimization techniques

---

## 6. Implementation Plan (Revised)

### Week 1: PyTorch Q² Core

**Day 1-2**: Implement Z₄, Z₈, Z₁₆ quantizers
```python
# Files to create:
q2_pytorch/
  quantizers.py    # Z₄, Z₈, Z₁₂, Z₁₆ classes
  layers.py        # Q2Linear, Q2Embedding
  utils.py         # Packing, Gray encoding
  test_quant.py    # Unit tests
```

**Day 3-4**: Integrate with standard transformer
```python
# Fork parameter-golf repo
# Replace nn.Linear with Q2Linear
# Verify training works at Z₁₆ (8-bit baseline)
```

**Day 5-7**: Progressive quantization
```python
# Implement Geode-guided training loop
# Test phase transitions
# Profile GPU utilization
```

### Week 2: Optimization

**Day 8-10**: Cache-line optimization
- Verify alignment with `torch.cuda.memory_stats()`
- Profile memory bandwidth utilization
- Optimize packing functions

**Day 11-14**: Hyperparameter tuning
- Learning rate sweep
- Layer depth vs. width trade-offs
- Z-ring allocation optimization

### Week 3-4: Competition Tuning

**Day 15-21**: Leaderboard chasing
- Match SOTA (1.14 bpb)
- Beat SOTA (target: 1.10 bpb)
- Reproducibility testing (5+ runs)

**Day 22-25**: Submission prep
- Package code
- Write documentation
- Submit PR

---

## 7. Why This Will Win

### 7.1 Novel Contributions

1. **First use of Z_N ring hierarchy** in Parameter Golf
2. **Cache-line optimized quantization** (no current entry does this)
3. **Geode-guided progressive training** (mathematically grounded)
4. **Power-of-2 bit widths** (stability + speed)

### 7.2 Expected Performance

| Metric | Current SOTA | Our Target | Advantage |
|--------|--------------|------------|-----------|
| Bits/byte | 1.1428 | 1.10 | 0.04 bpb |
| Training time | ~9.5 min | ~8 min | 1.5 min faster |
| Compressed size | ~15 MB | ~9 MB | 6 MB headroom |
| Inference speed | baseline | 1.3× faster | Better packing |

### 7.3 Risk Assessment

**Technical risks**: LOW
- Standard transformer architecture (proven)
- PyTorch native (no exotic dependencies)
- Power-of-2 bit widths (hardware-friendly)
- Multiple fallback options

**Timeline risk**: MEDIUM
- 3-4 weeks realistic
- Can submit incrementally (match SOTA first, then beat it)

**Competition risk**: MEDIUM
- Leaderboard active but slowing down
- Our approach is novel enough to be hard to replicate quickly

---

## 8. Comparison to Original Strategy

| Aspect | Original | Revised | Why Changed |
|--------|----------|---------|-------------|
| Implementation | WASM + PyTorch | Pure PyTorch | GPUs available, WASM unnecessary |
| Quantization | int5/int6 mixed | Z₄/Z₈/Z₁₂/Z₁₆ | Power-of-2 stability, ring structure |
| Architecture | LTC blocks | Standard attention | Proven baseline first |
| Training | Flat → progressive | Geode-guided | Explicit hierarchical training |
| Optimization | Generic | Cache-line aware | H100 specific optimizations |

**Key takeaway**: Original was overengineered. Revised is simpler, faster, and mathematically cleaner.

---

## 9. Next Steps

**Immediate** (today):
1. Create `q2_pytorch/` directory structure
2. Implement Z₄ quantizer with tests
3. Verify packing correctness

**This week**:
1. Complete all quantizers (Z₄, Z₈, Z₁₂, Z₁₆)
2. Integrate with forked parameter-golf repo
3. Train baseline Z₁₆ model

**Next week**:
1. Implement progressive training
2. Optimize for H100
3. First submission attempt

---

## 10. Acknowledgments

This revised strategy incorporates critical feedback on:
- Avoiding WASM for GPU-accelerated training
- Using power-of-2 bit widths for stability
- Exploring Z_N ring hierarchy
- Cache-line optimization for performance

The core Q² mathematical framework (Lee metric, Gray map, Geode factorization) remains valid. The implementation strategy is now properly aligned with the hardware constraints and mathematical structure of the problem.

---

**Document Status**: Ready for implementation
**Last Updated**: 2026-03-21
**Supersedes**: PARAMETER_GOLF_APPROACH.md
