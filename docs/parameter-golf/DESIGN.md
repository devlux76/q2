# Parameter Golf: Unified Design

> **Status**: Synthesized design from all prior analyses
> **Companion**: [APPROACH.md](APPROACH.md) · [code.py](code.py)
> **Mathematical foundations**: [DESIGN.md](../../DESIGN.md) (§D-x.y) · [WILDBERGER\_RUBINE\_REVIEW.md](WILDBERGER_RUBINE_REVIEW.md)

---

## Contents

1. [Thermodynamic Bounds](#1-thermodynamic-bounds)
2. [The Z₄ Quantization Kernel](#2-the-z4-quantization-kernel)
3. [Geode Architecture](#3-geode-architecture)
4. [CfC Blocks: Closed-Form Continuous-Time](#4-cfc-blocks-closed-form-continuous-time)
5. [GQA Blocks: Grouped Query Attention](#5-gqa-blocks-grouped-query-attention)
6. [Training Dynamics](#6-training-dynamics)
7. [Cache-Line and Register Geometry](#7-cache-line-and-register-geometry)
8. [Compression and Artifact Packing](#8-compression-and-artifact-packing)
9. [The DNA Isomorphism](#9-the-dna-isomorphism)

---

## 1 Thermodynamic Bounds

### 1.1 The information budget

The artifact has $B = 128{,}000{,}000$ bits. The training run produces
$T \approx 4.75 \times 10^{18}$ FLOP. By the Williams 2025 bound:

$$S_{\min} = \mathcal{O}\!\left(\sqrt{T \cdot \log_2 T}\right) \approx 1.72 \times 10^{10} \text{ bits}$$

We have $B / S_{\min} \approx 0.0075$ — less than 1% of the
information-theoretically implied storage. The model cannot faithfully encode
all structure discovered during training. It must **compress ruthlessly**.

**Design consequence**: every bit in the artifact must carry maximum
information. No padding, no odd-width alignment waste, no metadata overhead
that could be absorbed into the weight stream.

### 1.2 Inverting Williams: what can 16 MB encode?

$$B^2 \approx T_{\text{eff}} \cdot \log_2 T_{\text{eff}} \implies T_{\text{eff}} \approx 3.4 \times 10^{14} \text{ FLOP}$$

A 16 MB model encodes the structure of $\sim 3.4 \times 10^{14}$ FLOP — about
0.007% of the training budget. The remaining FLOP push stored structure toward
the FineWeb distribution without expanding capacity.

### 1.3 Optimal bit width from first principles

The question: what integer bit width $b$ maximizes $N = B / b$ (parameter
count) while achieving zero register/cache-line waste?

| $b$ | $N$ in 16 MB | Waste per 64-bit register | Ring structure | Verdict |
|:---:|:------------:|:-------------------------:|:--------------:|:--------|
| 1 | 128 M | 0 | $\mathbb{Z}_2$ (no complement) | Too coarse |
| **2** | **64 M** | **0** | **$\mathbb{Z}_4$ (full complement)** | **Optimal** |
| 4 | 32 M | 0 | $\mathbb{Z}_8$ | Viable fallback |
| 5 | ~24 M | 4 bits | None | Suboptimal |
| 6 | ~20 M | 4 bits | None | Suboptimal |
| 8 | 16 M | 0 | $\mathbb{Z}_{16}$ | Low capacity |

$b = 2$ uniquely satisfies: maximum $N$, zero waste, full $\mathbb{Z}_4$ ring
with complement involution, and Lee metric preserved by Gray encoding.

---

## 2 The Z₄ Quantization Kernel

### 2.1 The four cells

For a weight $w$ with per-row threshold $\tau^\ast$:

$$q(w) = \begin{cases}
A = 0 & w \leq -\tau^\ast & \text{strong negative (committed)} \\
B = 1 & -\tau^\ast < w \leq 0 & \text{weak negative (boundary)} \\
C = 2 & 0 < w \leq \tau^\ast & \text{weak positive (boundary)} \\
D = 3 & w > \tau^\ast & \text{strong positive (committed)}
\end{cases}$$

The threshold for Gaussian weights:

$$\tau^\ast = \frac{\Phi^{-1}(3/4)}{\sqrt{n}} \approx \frac{0.6745}{\sqrt{n}}$$

ensures equiprobable cells ($P(A) = P(B) = P(C) = P(D) = 1/4$), maximizing
entropy at $I = 2$ bits per dimension.

For non-Gaussian distributions (heavy-tailed activations, mixture models), the
threshold can alternatively be computed via the hyper-Catalan series
(Wildberger & Rubine 2025) — a combinatorial closed-form that converges
without iteration:

$$\alpha = \sum_\mathbf{m} C_\mathbf{m} \cdot t_2^{m_2} t_3^{m_3} \cdots$$

Truncation order trades precision for compute cost — a natural fit for the
resource-constrained setting.

### 2.2 Gray encoding

The Gray map $\phi: \mathbb{Z}_4 \to \mathbb{F}_2^2$:

$$g = s \oplus (s \gg 1)$$

| Symbol | Value | Gray code |
|:------:|:-----:|:---------:|
| A | 0 | 00 |
| B | 1 | 01 |
| C | 2 | 11 |
| D | 3 | 10 |

**Key property** (Hammons et al. 1994): Hamming distance on Gray codes equals
Lee distance on $\mathbb{Z}_4$ symbols:

$$d_{\text{Ham}}(\phi(u), \phi(v)) = d_{\text{Lee}}(u, v) = \sum_{i=1}^{n} \min(|u_i - v_i|, 4 - |u_i - v_i|)$$

This means Lee distance is computable via `popcnt(XOR)` — a single hardware
instruction on H100.

### 2.3 Complement involution

$$\theta(x) = x + 2 \pmod{4}: \quad A \leftrightarrow C, \quad B \leftrightarrow D$$

Properties:
- $\theta^2 = \text{id}$ (involution)
- $d_L(x, \theta(x)) = 2$ (maximum Lee distance)
- Encodes structural opposition (strong-negative ↔ weak-positive)

**Design role**: The complement constraint $\theta(W_{ij}) \neq W_{ij}$ for all
weights prevents redundant weight pairs, enforcing orthogonality at the symbolic
level. This acts as a **regularizer** during QAT.

### 2.4 Dequantization map

For the forward pass, symbols map to reconstruction centroids:

$$\hat{w}(s) = \{-1.5\tau, -0.5\tau, +0.5\tau, +1.5\tau\}[s]$$

The spacing is uniform in $\tau$-units. For non-Gaussian distributions, optimal
reconstruction uses the conditional expectation $\mathbb{E}[w \mid q(w) = s]$,
computable via hyper-Catalan series reversion (§5 of Wildberger-Rubine review).

### 2.5 Packing

Four symbols per byte, MSB-first:

```
byte = (g[4i] << 6) | (g[4i+1] << 4) | (g[4i+2] << 2) | g[4i+3]
```

32 weights per 64-bit register. 256 weights per 128-byte H100 cache line.
**Zero waste at every alignment boundary.**

---

## 3 Geode Architecture

### 3.1 The factorization

The Geode factorization of Q²'s transition sequences:

$$S(x) - 1 = \underbrace{4x}_{S_1} \cdot \underbrace{\frac{1}{1-3x}}_{G}$$

- $S_1 = 4x$: first symbol → 4 choices → **GQA block** (coarse context)
- $G = 1 + 3x + 9x^2 + \cdots$: each subsequent symbol → 3 choices → **CfC block** (refinement)

### 3.2 Layer layout

$$\underbrace{[\text{GQA},\ \text{CfC},\ \text{CfC},\ \text{CfC}]}_{\text{one Geode level}} \times 4 = 16 \text{ layers}$$

4 GQA + 12 CfC, ratio 3:1 (CfC:GQA). More CfC-heavy than LFM 2.5's
empirical 10:6 = 1.67:1 — predicted by the Geode for short-context (2048-token)
workloads where less attention is needed.

### 3.3 Information flow

| Depth | Layer type | Cumulative bits |
|:-----:|:-----------|:---------------:|
| 1 | GQA | 2.0 |
| 2–4 | CfC × 3 | 6.75 |
| 5 | GQA | 8.75 |
| 6–8 | CfC × 3 | 13.5 |
| 9 | GQA | 15.5 |
| 10–12 | CfC × 3 | 20.25 |
| 13 | GQA | 22.25 |
| 14–16 | CfC × 3 | 27.0 |

27 bits of structural information — within the 51.1-bit capacity of a full
32-symbol transition key (§D-3.6).

### 3.4 Euler polytope constraint

The hyper-Catalan coefficient governs admissible quantization lattices:

$$C_\mathbf{m} = \frac{(E-1)!}{(V-1)! \cdot \mathbf{m}!}, \quad V - E + F = 1$$

For $\mathbb{Z}_4$: $V = 4$, $E = 4$, $F = 1$ → $4 - 4 + 1 = 1$ ✓

This constrains the topology: we cannot add layers or heads arbitrarily.
Each architectural modification must preserve $V - E + F = \text{const}$,
which caps parameter growth and keeps the artifact under budget.

---

## 4 CfC Blocks: Closed-Form Continuous-Time

### 4.1 The LTC ODE

$$\dot{h}(t) = -\left[\frac{1}{\tau_c} + f(h, x; \theta)\right] h(t) + f(h, x; \theta)$$

### 4.2 Closed-form solution (Hasani et al. 2022)

$$h(t + \Delta t) = e^{-A_1 \Delta t} \odot h(t) + \frac{A_2}{A_1} \odot \left(1 - e^{-A_1 \Delta t}\right)$$

where $A_1, A_2$ are learned functions of $(x, h)$.

### 4.3 Parameter count

Per CfC layer with hidden dimension $d$:
- $A_1$ projection: $2d^2$ parameters (input + recurrent)
- $A_2$ projection: $2d^2$ parameters
- Output projection: $d^2$ parameters
- **Total: $5d^2$ per CfC block**

Compare to GQA: $\approx 11.67d^2$ per block. CfC is **2.3× more parameter-efficient**.

### 4.4 Q² synergy

CfC state updates use sigmoid activations that saturate at $\pm 1$. Near
saturation, exact weight values matter less than **sign and magnitude class** —
precisely what Z₄ preserves.

The two matrices $A_1$ (decay) and $A_2$ (integration) have a natural
**complement relationship**: strong-decay and strong-integration are complements
in the same way that $A$ and $C$ are complements in $\mathbb{Z}_4$.

### 4.5 Implementation

```python
class CfCBlock(nn.Module):
    """One Geode G-level: 3-way refinement via closed-form LTC."""

    def __init__(self, d_model, n_time_constants=5):
        super().__init__()
        self.a1_proj = Q2Linear(d_model, d_model)  # Decay
        self.a2_proj = Q2Linear(d_model, d_model)  # Integration
        self.out_proj = Q2Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.randn(n_time_constants))
        self.ln = nn.LayerNorm(d_model)
        # SwiGLU MLP
        self.mlp_up = Q2Linear(d_model, d_model * 3)
        self.mlp_gate = Q2Linear(d_model, d_model * 3)
        self.mlp_down = Q2Linear(d_model * 3, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, h):
        # CfC state update
        x_norm = self.ln(x)
        a1 = torch.sigmoid(self.a1_proj(x_norm))
        a2 = torch.sigmoid(self.a2_proj(x_norm))
        tau_c = torch.sigmoid(self.tau)
        h_new = torch.exp(-a1 * tau_c) * h + (a2 / a1) * (1 - torch.exp(-a1 * tau_c))
        x = x + self.out_proj(h_new)
        # SwiGLU MLP
        x = x + self.mlp_down(F.silu(self.mlp_gate(self.ln2(x))) * self.mlp_up(self.ln2(x)))
        return x, h_new
```

---

## 5 GQA Blocks: Grouped Query Attention

### 5.1 Role in Geode architecture

GQA blocks are the **$S_1$ coarse selectors** — they attend across the full
sequence to establish broad context structure (equivalent to selecting one of
4 block files in the transition key, §D-3.4).

### 5.2 Implementation

Standard Grouped Query Attention with:
- $n_h$ query heads, $n_{\text{kv}}$ key-value heads ($n_h / n_{\text{kv}}$ groups)
- All projections (Q, K, V, O) are `Q2Linear` (Z₄ quantized)
- SwiGLU MLP with 3× expansion, all `Q2Linear`
- Uses `F.scaled_dot_product_attention` → FlashAttention-2 kernel on H100

```python
class GQABlock(nn.Module):
    """One Geode S1-level: 4-way coarse selection via grouped query attention."""

    def __init__(self, d_model, n_heads=8, n_kv_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = Q2Linear(d_model, d_model)
        self.k_proj = Q2Linear(d_model, self.head_dim * n_kv_heads)
        self.v_proj = Q2Linear(d_model, self.head_dim * n_kv_heads)
        self.o_proj = Q2Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        # SwiGLU MLP
        self.mlp_up = Q2Linear(d_model, d_model * 3)
        self.mlp_gate = Q2Linear(d_model, d_model * 3)
        self.mlp_down = Q2Linear(d_model * 3, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln1(x)
        q = self.q_proj(h).view(*h.shape[:-1], self.n_heads, self.head_dim)
        k = self.k_proj(h).view(*h.shape[:-1], self.n_kv_heads, self.head_dim)
        v = self.v_proj(h).view(*h.shape[:-1], self.n_kv_heads, self.head_dim)
        # GQA: repeat KV heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=-2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=-2)
        # Transpose for attention: (B, H, T, D)
        q, k, v = [t.transpose(-3, -2) for t in (q, k, v)]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(-3, -2).contiguous().view(*x.shape)
        x = x + self.o_proj(attn)
        # SwiGLU MLP
        h2 = self.ln2(x)
        x = x + self.mlp_down(F.silu(self.mlp_gate(h2)) * self.mlp_up(h2))
        return x
```

---

## 6 Training Dynamics

### 6.1 QAT via STE

During training, each `Q2Linear` layer:
1. Quantizes weights to $\{A, B, C, D\}$
2. Dequantizes to reconstruction centroids for the forward pass
3. Passes gradients through via STE (straight-through estimator)
4. Updates full-precision shadow weights with the optimizer

The FP32 warm-up phase (10% of training) establishes activation distributions
before imposing the Z₄ constraint. This follows the BitNet finding (Ma et al.
2024) that QAT-from-scratch requires a brief float-precision warm-up.

### 6.2 Progressive quantization (Geode-guided)

Layers are quantized in Geode order: deep CfC layers first (most tolerant of
low precision), then middle layers, then GQA layers, then embedding adjacent.

This matches the Geode's hierarchical decomposition: coarse structure ($S_1$)
is established first, then refinement ($G$) is progressively constrained.

### 6.3 Muon optimizer

Nesterov momentum with per-matrix spectral normalization:
- Prevents large weight moves from disrupting Q² complement structure
- Higher LR (0.01) than Adam due to Nesterov momentum
- Weight decay 0.04 on matrices only

### 6.4 Stochastic weight averaging

SWA activated from 60% of training. The averaged model produces smoother
loss landscapes that are more amenable to 2-bit quantization — flat minima
tolerate quantization error better than sharp minima.

---

## 7 Cache-Line and Register Geometry

### 7.1 H100 memory hierarchy

| Level | Size | Access time | Alignment |
|:------|:-----|:------------|:----------|
| Register file (per SM) | 256 KB | 1 cycle | 32-bit |
| L1/shared memory (per SM) | 228 KB | ~28 cycles | 128-byte |
| L2 cache (per GPU) | 50 MB | ~200 cycles | 128-byte |
| HBM3 | 80 GB | ~400 cycles | 128-byte |

### 7.2 Z₄ alignment at every level

| Alignment boundary | Size | Z₄ weights fitting | Waste |
|:-------------------|:-----|:-------------------:|:-----:|
| 32-bit register | 4 B | 16 | 0 |
| 64-bit double-word | 8 B | 32 | 0 |
| 128-byte cache line | 128 B | 512 | 0 |
| 256-byte aligned block | 256 B | 1024 | 0 |

Z₄ achieves **perfect alignment at every level** of the H100 memory hierarchy.
int5 wastes 4 bits per 64-bit word, accumulating to 1 MB of waste across 16 MB.

### 7.3 Tensor dimension constraints

To ensure perfect cache-line alignment, all tensor dimensions must be
divisible by 512 (weights per cache line) or at minimum 32 (weights per
register). With $d = 768$:
- $768 = 32 \times 24$ ✓ (register-aligned)
- $768 \times 3 = 2304 = 32 \times 72$ ✓ (MLP expansion)

### 7.4 LIV cache-line packing (optional)

For post-training int5 export (LFM 2.5 compatibility):

12 LIV symbols × 5 bits + 2-bit Q² tag + 2 unused = 64 bits exactly.

The Q² tag partitions packed words into 4 groups for parallel SM dispatch.
The top 10 × 5 = 50 bits form two 5 × 5 binary matrices whose Boolean
product serves as a codon checksum — verifiable in $O(25)$ bitwise ops.

---

## 8 Compression and Artifact Packing

### 8.1 Q2BN binary format

The Q2BN format stores quantized weights:

```
[4-byte magic: "Q2BN"]
[4-byte version]
[4-byte tensor count]
For each tensor:
  [4-byte name length][name bytes]
  [4-byte ndim][4-byte × ndim shape]
  [4-byte dtype: 0=Q2, 1=FP16, 2=FP32]
  [packed weight bytes]
```

### 8.2 Geode-ordered serialization

Tensors are serialized in Geode traversal order:
1. GQA block 1 weights (all projections)
2. CfC blocks 2–4 weights
3. GQA block 5 weights
4. CfC blocks 6–8 weights
5. ... (repeat pattern)

This ordering groups structurally similar weights together, producing long
runs of similar byte patterns that zstd exploits for higher compression.

### 8.3 Compression pipeline

```python
# 1. Pack weights to Q2BN
q2_pack.pack_state_dict(model.state_dict(), 'model.q2bin')

# 2. Compress with zstd level 22
import zstandard
cctx = zstandard.ZstdCompressor(level=22)
compressed = cctx.compress(open('model.q2bin', 'rb').read())

# 3. Validate
assert len(compressed) <= 16_000_000
```

---

## 9 The DNA Isomorphism

### 9.1 Nature's billion-year head start

The choice of $\mathbb{Z}_4$ is not arbitrary. DNA uses four bases:

| DNA | Q² | Binary | Complement |
|:---:|:--:|:------:|:----------:|
| A (Adenine) | A (strong −) | 00 | T ↔ C |
| C (Cytosine) | B (weak −) | 01 | G ↔ D |
| G (Guanine) | C (weak +) | 11 | C ↔ A |
| T (Thymine) | D (strong +) | 10 | A ↔ B |

The complement pairing (A↔T, C↔G in DNA; A↔C, B↔D in Q²) is the same
involution $\theta(x) = x + 2 \bmod 4$.

### 9.2 Codons as Geode levels

DNA codons are triplets of bases: $4^3 = 64$ possible codons encoding 20
amino acids. This is the Geode's 3-way refinement at each level:

$$G = 1 + 3x + 9x^2 + 27x^3 + \cdots$$

At depth 3: $4 \times 3^2 = 36$ distinct run-reduced sequences — close to
the 20 amino acids when accounting for redundancy (the "wobble" in the third
codon position).

### 9.3 What this means for parameter golf

Nature evolved $\mathbb{Z}_4$ as the optimal encoding for information in a
thermodynamically constrained environment. The parameter golf challenge
presents the same problem: encode maximum information (language structure)
in minimum space (16 MB) under fixed compute (10 minutes × 8 × H100).

The isomorphism is not metaphor — it is structural. The same mathematics
(Gray encoding, Lee metric, complement involution, Geode factorization) that
describes DNA coding theory describes our weight quantization scheme.

We are not borrowing a biological metaphor. We are recognizing that both
problems — storing heritable information in nucleotides and storing linguistic
structure in quantized weights — are instances of the same $\mathbb{Z}_4$
optimization under resource constraints.

---

## References

- Williams, R. (2025). *Simulating Time With Square-Root Space*. Proc. STOC 2025. arXiv:2502.17779.
- Wildberger, N. J. & Rubine, D. (2025). *A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode*. Amer. Math. Monthly 132:5, 383–402.
- Hammons, A. R. et al. (1994). *The $\mathbb{Z}_4$-linearity of Kerdock, Preparata, Goethals, and related codes*. IEEE Trans. Inform. Theory 40:2, 301–319.
- Hasani, R. et al. (2021). *Liquid Time-constant Networks*. AAAI-2021. arXiv:2006.04439.
- Hasani, R. et al. (2022). *Closed-form Continuous-time Neural Networks*. Nature Machine Intelligence 4, 992–1003. arXiv:2106.13898.
- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. arXiv:2402.12263.
- Liquid AI. *LFM 2.5 Technical Report* (2025). https://www.liquid.ai/research/lfm-2-5
- OpenAI. *Parameter Golf*. https://openai.com/index/parameter-golf/
