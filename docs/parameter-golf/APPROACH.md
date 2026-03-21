# Parameter Golf: Unified Approach

> **Status**: Synthesis of all prior analyses — the definitive strategy
> **Source documents**: [ANALYSIS.md](ANALYSIS.md) · [APPROACH_INITIAL.md](APPROACH_INITIAL.md) · [APPROACH_REVISED.md](APPROACH_REVISED.md) · [STRATEGY.md](STRATEGY.md) · [IMPLEMENTATION.md](IMPLEMENTATION.md) · [WILDBERGER_RUBINE_REVIEW.md](WILDBERGER_RUBINE_REVIEW.md)
> **Related**: [DESIGN.md](DESIGN.md) · [code.py](code.py)

---

## 0 Starting from constraints

Every correct solution starts from what is **known and fixed**, not from what
others did. The constraints determine the solution; the solution does not choose
its constraints.

### 0.1 Hard constraints

| Constraint | Value | Bits |
|:-----------|:------|:-----|
| Artifact size | 16,000,000 bytes | 128,000,000 bits |
| Training wall-clock | 600 seconds | — |
| Hardware | 8 × H100 SXM | — |
| Metric | val\_bpb on FineWeb (tokenizer-agnostic) | lower is better |

### 0.2 Hardware knowns (H100 SXM)

| Resource | Per GPU | 8 × GPU |
|:---------|:--------|:--------|
| BF16 tensor-core FLOPS | 989 TFLOPS | 7,912 TFLOPS |
| L2 cache | 50 MB | 400 MB total |
| HBM3 bandwidth | 3.35 TB/s | 26.8 TB/s |
| HBM3 capacity | 80 GB | 640 GB |
| SM count | 132 | 1,056 |
| Register file per SM | 256 KB | — |
| Shared memory per SM | 228 KB | — |
| Cache line | 128 bytes | — |
| CUDA register width | 32 bits | — |
| Max warps per SM | 64 | — |
| NVLink bandwidth | 900 GB/s | — |

**Total available compute** in 10 minutes:

$$T = 8 \times 989 \times 10^{12} \times 600 \approx 4.75 \times 10^{18} \text{ FLOP}$$

### 0.3 Williams 2025 SpaceTime bound

Ryan Williams proved (STOC 2025, arXiv:2502.17779) that any computation
running in time $t$ can be simulated in space:

$$S = \mathcal{O}\!\left(\sqrt{t \cdot \log t}\right)$$

Applied to our constraints:

$$S_{\min} = \sqrt{4.75 \times 10^{18} \times 62} \approx 1.72 \times 10^{10} \text{ bits} \approx 2.15 \text{ GB}$$

Our artifact provides $1.28 \times 10^8$ bits — **0.75% of the
Williams-implied storage**. This means:

1. We are in a **deep-compression regime** — every bit is precious.
2. Only the most structured, compressible patterns in FineWeb can be captured.
3. The model stores $\sim 3.4 \times 10^{14}$ FLOP of effective computation — the
   remaining training FLOP refine weights toward the target distribution without
   encoding qualitatively new structure.
4. **Any format that wastes bits (padding, metadata, odd-width alignment)
   directly increases bpb.**

### 0.4 The Wildberger–Geode result

The Geode factorization (Wildberger & Rubine 2025):

$$S - 1 = S_1 \cdot G$$

decomposes every non-trivial discrete structure into:

- **$S_1$**: the coarse first-level choice (4 ways for $\mathbb{Z}_4$)
- **$G = 1/(1-3x)$**: the refinement (3 choices per subsequent step)

This is not metaphor — it is isomorphic to:
- **DNA**: 4 bases ($\mathbb{Z}_4$), codons (triplets of 3-choice refinements)
- **Q² transition trie**: root arity 4, subsequent arity 3
- **Progressive quantization**: coarse cell → refinement within cell

The factorization provides the architectural template: **[coarse, refine, refine, refine] repeated**.

### 0.5 The $\mathbb{Z}_4$ optimality

Nature runs on $\mathbb{Z}_4$. DNA uses 4 bases: {A, C, G, T}. This is not
coincidence — it is the minimum alphabet that simultaneously preserves:

1. **Sign** (which side of a hyperplane)
2. **Magnitude class** (near boundary or committed)
3. **Complement structure** (A↔T, C↔G; in Q²: $\theta(x) = x + 2 \bmod 4$)

At 2 bits per symbol, $\mathbb{Z}_4$ quantization:
- Packs **32 weights per 64-bit register** — zero waste
- Packs **256 weights per 128-byte H100 cache line** — zero waste
- Achieves **$N = 64$ M parameters** in 16 MB — 2.8× more than int5 SOTA
- Preserves Lee metric distances via Gray encoding ($d_L = \text{popcnt}(\text{XOR})$)

Compare to the current SOTA (int5):
- 12 weights per 64-bit register, **4 bits wasted per register**
- Across 16 MB: 1 MB of pure waste ($\approx 4$ M lost $\mathbb{Z}_4$ parameters)
- Only ~24 M effective parameters vs our 64 M

---

## 1 What convergence tells us

Four independent analyses arrived at these common conclusions:

| Finding | Analyses agreeing | Confidence |
|:--------|:-----------------:|:----------:|
| Power-of-2 bit widths beat odd widths | ANALYSIS, APPROACH\_REVISED, Williams | High |
| Geode-guided progressive training beats flat training | ANALYSIS, STRATEGY, APPROACH\_INITIAL | High |
| CfC/LTC blocks are more parameter-efficient than attention | ANALYSIS, APPROACH\_INITIAL, STRATEGY | High |
| BigramHash tokenizer is optimal at 10k vocab | All four | High |
| Pure PyTorch on GPU, no WASM | APPROACH\_REVISED, STRATEGY | High |
| Mixed-precision: high bits for embedding, low bits for deep layers | APPROACH\_REVISED, STRATEGY | Medium |

Where analyses **diverge**, we take the strongest position:

| Divergence | Resolution | Rationale |
|:-----------|:-----------|:----------|
| int5/int6 vs Z₄ 2-bit | **Z₄ 2-bit** | Williams + cache alignment + 2.8× more params |
| 12 layers × 384 dim vs 16 layers × 768 dim | **16 layers × 768 dim** | Z₄ budget allows 64M params; use them |
| Standard attention vs full CfC | **Hybrid [GQA, CfC, CfC, CfC] × 4** | Geode-derived; GQA for coarse context, CfC for refinement |
| Uniform vs hierarchical Z-ring | **Uniform Z₄** | Maximizes N; Z₈/Z₁₆ only for embedding if needed |

---

## 2 The architecture

### 2.1 Geode-derived layout: [GQA, CfC, CfC, CfC] × 4

From the Geode factorization $S_1 = 4x$ (coarse) and $G = 1/(1-3x)$ (refine):

| Layer | Type | Geode role | Information gain |
|:-----:|:-----|:-----------|:-----------------|
| 1 | GQA | $S_1$ root | $\log_2 4 = 2$ bits coarse context |
| 2–4 | CfC × 3 | $G$ level 1 | $3 \times \log_2 3 \approx 4.75$ bits refinement |
| 5 | GQA | $S_1$ reset | Re-establishes coarse context |
| 6–8 | CfC × 3 | $G$ level 2 | Refinement |
| 9 | GQA | $S_1$ reset | Re-establishes coarse context |
| 10–12 | CfC × 3 | $G$ level 3 | Refinement |
| 13 | GQA | $S_1$ reset | Final coarse context |
| 14–16 | CfC × 3 | $G$ level 4 | Final refinement |

**Total structural capacity**: $4 \times (2 + 3 \times 1.585) \approx 27$ bits —
within the 51.1-bit capacity of the full 32-symbol key.

### 2.2 Parameter budget

With $d = 768$, $n_{\text{kv}} = 4$ KV heads, MLP ratio 3×:

| Component | Formula | Parameters | Storage (Z₄) |
|:----------|:--------|:----------:|:-------------:|
| Embedding (V=1024, tied) | $1024 \times 768 \times 2$ | 1.57 M | 1.57 MB (FP16) |
| 4 × GQA block | $4 \times 11.67 d^2$ | 27.5 M | 6.88 MB |
| 12 × CfC block | $12 \times 5 d^2$ | 35.4 M | 8.85 MB |
| LayerNorm (16 layers) | negligible | ~25 K | ~50 KB (FP16) |
| **Total** | | **~64.5 M** | **~17.3 MB raw** |

After zstd-22 compression (conservative 0.85×): **~14.7 MB** — within budget
with 1.3 MB headroom.

If too tight, reduce $d$ to 700–730 or use $V = 256$ (byte tokenization,
saving 1.2 MB on embedding).

### 2.3 Byte tokenization option

At the byte level, vocabulary is always exactly 256:

| Tokenization | Vocab | Embedding cost | Tokenizer |
|:-------------|:-----:|:--------------:|:---------:|
| SP-1024 | 1,024 | 1.57 MB (FP16) | Required |
| BigramHash 10240 | 10,240 | ~15.7 MB | Required |
| Raw bytes | 256 | 0.39 MB (FP16) | **None** |

Byte tokenization frees ~1.2 MB vs SP-1024 ($\approx 5$ M extra Z₄ weights)
and eliminates the tokenizer encoder entirely. FineWeb bpb scoring operates on
bytes, so there is no evaluation penalty.

---

## 3 The quantization

### 3.1 Z₄ structural quantization

All linear weight matrices $W \in \mathbb{R}^{m \times n}$ are quantized to
$\{A, B, C, D\} = \{0, 1, 2, 3\} \subset \mathbb{Z}_4$:

$$q(w) = \begin{cases}
A & w \leq -\tau^\ast \\
B & -\tau^\ast < w \leq 0 \\
C & 0 < w \leq \tau^\ast \\
D & w > \tau^\ast
\end{cases}$$

where $\tau^\ast = \Phi^{-1}(3/4) / \sqrt{n} \approx 0.6745 / \sqrt{n}$.

**Gray encoding**: $g = s \oplus (s \gg 1)$ maps symbols so that
$d_{\text{Hamming}}(g_i, g_j) = d_{\text{Lee}}(s_i, s_j)$.

**Packing**: 4 symbols per byte, MSB-first:

```
byte = (g[4i] << 6) | (g[4i+1] << 4) | (g[4i+2] << 2) | g[4i+3]
```

### 3.2 Why Z₄ beats reconstruction quantization

| Property | Reconstruction (GPTQ/int5) | Structural (Q²/Z₄) |
|:---------|:---------------------------|:--------------------|
| Objective | $\min \lVert W - \hat{W} \rVert_F^2$ | Preserve relational geometry |
| Bits/weight | 5–6 | **2** |
| Params in 16 MB | ~24 M | **~64 M** |
| Register waste | 4 bits/register | **0** |
| Ring structure | None | $\mathbb{Z}_4$ with Lee metric |
| Complement | None | $\theta(x) = x + 2 \bmod 4$ |
| Gray encoding | N/A | Hamming = Lee distance |

### 3.3 Straight-through estimator for QAT

The STE propagates gradients through quantization:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}} \approx \frac{\partial \mathcal{L}}{\partial \hat{W}_{ij}} \cdot \mathbf{1}\!\left[|W_{ij}| \leq \kappa\right]$$

with passthrough window $\kappa = 3\tau^\ast$.

Threshold $\tau^\ast$ is refreshed every 1024 steps from the empirical 25th/75th
percentile of each weight row (reservoir calibration, §D-2.5).

### 3.4 Precision allocation

| Component | Precision | Rationale |
|:----------|:----------|:----------|
| Embedding | FP16 | Interface between tokens and continuous space; small (V=256 or 1024) |
| GQA projections (Q, K, V, O) | Z₄ (2-bit) | Coarse context; complement structure natural |
| GQA MLP (up, gate, down) | Z₄ (2-bit) | Bulk of parameters; Z₄ maximizes N |
| CfC state matrices ($A_1$, $A_2$) | Z₄ (2-bit) | Complement structure ($A_1$ decay ↔ $A_2$ integration) |
| LayerNorm γ, β | FP16 | Negligible count; critical for stability |

---

## 4 The training strategy

### 4.1 Three-phase Geode-guided training

**Phase 1 — FP32 warm-up (60 seconds, 10% of budget)**

Train the full model at FP32 to establish activation distributions before
imposing the Z₄ constraint. OrthoInit for GQA, Kaiming for CfC/MLP.
Freeze embeddings for the first 500 steps to stabilize hash collisions.

**Phase 2 — Q²-QAT progressive quantization (360 seconds, 60% of budget)**

Activate Z₄ quantization layer-by-layer following the Geode hierarchy:
deep layers first (they tolerate 2-bit best), then middle, then shallow.
Each layer: quantize → fine-tune → proceed.

Enable SWA (stochastic weight averaging) from step 60%.

**Phase 3 — Final refinement (180 seconds, 30% of budget)**

All layers at Z₄. Cosine LR cooldown. Final SWA pass with weight decay 0.04.
Sliding-window evaluation (stride 64) to harvest lower bpb.

### 4.2 Optimizer and schedule

| Setting | Value | Source |
|:--------|:------|:-------|
| Optimizer | Muon (Nesterov + spectral norm) | Leaderboard SOTA |
| Learning rate | 0.01 (cosine with warmup 200 steps) | Leaderboard SOTA |
| Weight decay | 0.04 (matrices only) | Leaderboard SOTA |
| SWA | Last 40% of training | Leaderboard SOTA |
| Gradient clipping | 1.0 | Training stability |
| Sequence length | 2048 (Phase 1–2), 4096 (Phase 3) | Context scaling |
| Q² threshold refresh | Every 1024 steps | §D-2.5 |

### 4.3 H100 optimizations

```python
# BF16 for non-quantized operations (H100 native)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

# Compile for max throughput
model = torch.compile(model, mode='max-autotune')

# FlashAttention for GQA blocks
# F.scaled_dot_product_attention uses FlashAttention-2 on H100

# CfC blocks: element-wise sigmoid/multiply — no FlashAttention overhead
```

### 4.4 Data pipeline

8 × H100 data-parallel with gradient accumulation:

```python
effective_batch_tokens = batch_per_gpu * seq_len * 8 * grad_accum
# Target: ~4M tokens per optimizer step
# With seq_len=2048, batch=32, grad_accum=4: 32 * 2048 * 8 * 4 ~ 2M tokens
```

---

## 5 Artifact packaging

### 5.1 Export pipeline

1. Select SWA-averaged checkpoint
2. Pack all weight matrices to Q2BN format (Gray-encoded, 4 symbols/byte)
3. Order tensors by Geode traversal (long runs → RLE-friendly for zstd)
4. Compress with zstd level 22
5. Validate: total artifact ≤ 16,000,000 bytes

### 5.2 Artifact structure

```
Header (1 KB):
  Model config, quantization thresholds per layer, vocabulary

Body (~14 MB):
  Embedding (FP16, ~0.4 MB for V=256)
  GQA weights (Z4 packed, ~6.9 MB)
  CfC weights (Z4 packed, ~8.9 MB)
  LayerNorm parameters (FP16, ~50 KB)

Total before zstd: ~16.3 MB
After zstd-22 (~0.85×): ~13.8 MB
Headroom: ~2.2 MB
```

---

## 6 Performance projection

### 6.1 Scaling law

Under Chinchilla scaling ($\alpha \approx 0.34$, $A \approx 406.4$):

$$\Delta L \approx A \cdot (N_{24M}^{-0.34} - N_{64M}^{-0.34}) \approx 0.056 \text{ nats} \approx 0.081 \text{ bpb}$$

### 6.2 Projected performance

| Component | Estimated bpb gain |
|:----------|:------------------:|
| Current SOTA baseline | 1.1428 |
| Z₄ parameter scaling ($2.8\times N$) | −0.08 |
| CfC architecture efficiency | −0.02 to −0.05 |
| Geode-guided progressive training | −0.01 |
| Zero-waste cache-line alignment | −0.005 |
| **Projected total** | **~1.00 to 1.05** |

### 6.3 Risk-adjusted estimate

Conservative (only scaling benefit works): **1.06 bpb**
Expected (scaling + architecture): **1.03 bpb**
Optimistic (all innovations compound): **1.00 bpb**

Any of these substantially beat the current SOTA of 1.1428 bpb.

---

## 7 Execution

### 7.1 Immediate

1. Implement Z₄ quantizer with Gray encoding and STE
2. Implement CfC block and GQA block (all projections use Q2Linear)
3. Assemble 16-layer Geode model
4. Single-GPU smoke test (200 steps)

### 7.2 This week

1. 8×H100 full training run (10 minutes)
2. Validate compressed artifact size
3. First bpb measurement on FineWeb validation

### 7.3 Iterate

1. Tune $d$, $V$, sequence length, LR, weight decay
2. Ablate: CfC vs attention, byte vs SP-1024, progressive vs flat
3. Target reproducibility: 5+ runs within σ < 0.005 bpb

---

## References

- Williams, R. (2025). *Simulating Time With Square-Root Space*. STOC 2025. arXiv:2502.17779.
- Wildberger, N. J. & Rubine, D. (2025). *A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode*. Amer. Math. Monthly 132:5, 383–402.
- Hammons, A. R. et al. (1994). *The $\mathbb{Z}_4$-linearity of Kerdock, Preparata, Goethals, and related codes*. IEEE Trans. Inform. Theory 40:2, 301–319.
- Hasani, R. et al. (2021). *Liquid Time-constant Networks*. AAAI-2021.
- Hasani, R. et al. (2022). *Closed-form Continuous-time Neural Networks*. Nature Machine Intelligence 4, 992–1003.
- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. arXiv:2402.12263.
- OpenAI. *Parameter Golf*. https://openai.com/index/parameter-golf/
