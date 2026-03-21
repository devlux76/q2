# Parameter Golf: A Q²-Based Strategy

> **Related documents:** [DESIGN.md](DESIGN.md) · [RELATED_WORK.md](RELATED_WORK.md)

Section references of the form §D-x.y refer to [DESIGN.md](DESIGN.md).
Section references of the form §R-x refer to [RELATED_WORK.md](RELATED_WORK.md).

---

## Contents

1. [The Challenge](#1-the-challenge)
2. [Current State of the Art](#2-current-state-of-the-art)
3. [The Q² Compression Advantage](#3-the-q-compression-advantage)
4. [Architecture: Liquid Time Constant Networks](#4-architecture-liquid-time-constant-networks)
   - 4.5 [Geode-derived layer layout](#45-geode-derived-layer-layout)
5. [The Combined Strategy](#5-the-combined-strategy)
   - 5.5 [LIV cache-line packing and byte tokenization](#55-liv-cache-line-packing-and-byte-tokenization)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Performance Projections](#7-performance-projections)
   - 7.5 [Williams SpaceTime bound and optimal bit width](#75-williams-spacetime-bound-and-optimal-bit-width)
8. [References](#references)

---

## 1 The Challenge

OpenAI's **Parameter Golf** challenge (March–April 2026) asks participants to train
the language model that achieves the lowest bits-per-byte (bpb) on the FineWeb
validation set, subject to:

1. **Artifact size:** total compressed artifact (code + compressed model weights) ≤
   16,000,000 bytes (decimal 16 MB).
2. **Training time:** ≤ 10 minutes on 8×H100 SXM GPUs.
3. **Evaluation:** tokenizer-agnostic bpb on the first 50 000 FineWeb documents.

This is a form of *L(N)* optimisation in neural scaling-law notation — minimise
loss given a fixed parameter budget — unconstrained by data or total compute, but
tightly constrained by artifact size and training speed.

The challenge is inspired by NanoGPT Speedrunning (L(T) optimisation) and
NanoGPT Slowrun (L(D) optimisation). All three are special cases of the same
Pareto frontier: the scaling law surface $L(N, D, T)$.

---

## 2 Current State of the Art

The top leaderboard entries as of March 2026 use a consistent set of techniques:

| Run | bpb | Key techniques |
|:----|:---:|:---------------|
| 10L Int5-MLP + BigramHash(10240) | 1.1428 | Int5/Int6 mixed QAT, BigramHash, SWA 0.4, WD=0.04 |
| Int6 MLP3x + SmearGate + BigramHash | 1.1458 | Int6 QAT, 3x MLP, SmearGate, OrthoInit, SWA |
| 11L MLP3x + Int6 QAT | 1.1502 | 11 layers, 3x MLP, Int6 QAT, zstd-22, sliding eval |
| Naive Baseline | 1.2244 | 9 layers, 512 dim, 1024 vocab, tied embeddings |

The parameter budget for current SOTA entries is approximately:

$$N_{\text{SOTA}} \approx \frac{(B - C) \cdot 8}{b_{\text{eff}}}$$

where $B = 16 \times 10^6$ bytes is the total budget, $C \approx 50{,}000$ bytes
is the code footprint, and $b_{\text{eff}} \approx 5.5$ is the effective bits per
weight after int5/int6 packing and zstd-22 compression:

$$N_{\text{SOTA}} \approx \frac{(16{,}000{,}000 - 50{,}000) \times 8}{5.5} \approx 23 \text{ M parameters}$$

The BigramHash technique partitions the 16 MB budget between a vocabulary bigram
table (providing a strong unigram/bigram prior cheaply) and the neural model
(providing long-range context). The best entries use a vocabulary of 1024–10240
tokens; at 1024 tokens a complete bigram table costs $1024^2 \times 1 \approx 1$ MB,
leaving ~15 MB for the neural model.

**What the current SOTA does not do:**
- It does not use sub-5-bit structural quantization designed for maximum
  information preservation per bit (§D-2.4).
- It does not use recurrent or state-space architectures that provide sequential
  memory without O(n²) attention cost.
- It does not exploit the complement structure of the $\mathbb{Z}_4$ alphabet
  (§D-2.8) as an inductive bias for weight organisation.

---

## 3 The Q² Compression Advantage

### 3.1 Parameter capacity at 2 bits

Q² uses 2 bits per symbol, packing 4 symbols per byte. Applied to model weights
as a quantization-aware training (QAT) scheme — training with the quaternary
constraint from the start, as BitNet does with ternary weights (§R-3.1) — the
parameter capacity in 16 MB is:

$$N_{\text{Q}^2} \approx \frac{(B - C) \cdot 8}{2} \approx \frac{15{,}950{,}000 \times 8}{2} \approx 63.8 \text{ M parameters}$$

This is a **2.8× increase** in parameter count at the same artifact size, relative
to the current int5/int6 SOTA.

If the Q² weights compress by an additional factor of $r$ under zstd-22 (possible
when trained weights exhibit run-length structure that Q²'s Gray encoding exploits,
§D-2.7), the capacity grows further:

$$N_{\text{Q}^2,\, r} \approx 63.8 \cdot r \text{ M parameters}$$

For $r = 1.2$ (conservative 20% compression beyond raw 2-bit packing), the
effective capacity is ~76 M parameters.

### 3.2 Why structural quantization outperforms uniform grids at 2 bits

Standard int2 post-training quantization (GPTQ/AWQ at 2 bits) loses substantially
more accuracy than int4 because the reconstruction objective:

$$\min_{\hat{W}} \| W - \hat{W} \|_F^2$$

tries to approximate float32 weights with 4 levels, and the quantization error at
2 bits is large enough to disrupt learned representations.

Q² structural quantization has a different objective: preserve the *relational
geometry* of the weight space, not the pointwise values. The four cells
$\{A, B, C, D\}$ encode **sign** and **magnitude class**, which are the two
structural features that determine a weight's contribution to the L1 geometry of
activation space (§D-1.5). A weight quantized to $A$ (strong negative) and one
quantized to $C$ (weak positive) are separated by Lee distance 2 — the complement
distance — reflecting a fundamental opposition in their role, not an accident of
the numerical grid.

This matters for QAT because:

1. **Complement involution as a regulariser.** The constraint $\theta(W_{ij}) \neq W_{ij}$
   for all weights (§D-2.8) prevents the model from learning redundant weight pairs
   where $W_{ij}$ and $W_{kl}$ encode the same functional direction. It enforces
   orthogonality of the weight organisation at the symbolic level.

2. **Lee metric loss.** Training with a Lee distance penalty on weight changes
   encourages the model to make transitions that preserve complement structure.
   Gradient steps that would move $A \to C$ (complement flip, Lee distance 2) are
   penalised more than steps that move $A \to B$ (adjacent, Lee distance 1).

3. **Gray encoding preserves gradient flow.** The Gray map $\phi$ (§D-2.7) makes
   Hamming distance on the encoded bits equal to Lee distance on the symbols.
   The straight-through estimator (STE) for Q²-QAT propagates gradients through
   the Gray encoding as if the quantization were a smooth threshold operation,
   and the bit-level gradient is correctly ordered: a gradient pointing from $A$
   toward $D$ passes through $B$ and $C$ in order, not by a shortcut.

### 3.3 Expected compression benefit

The Gray-encoded weight tensor of a Q²-trained model has a specific statistical
structure. After training, the equiprobable condition (§D-2.5):

$$P(W_{ij} = A) = P(W_{ij} = B) = P(W_{ij} = C) = P(W_{ij} = D) = \tfrac{1}{4}$$

is the maximum-entropy condition: all four symbols are equally likely, so the raw
2-bit stream is nearly incompressible. The compression ratio $r \approx 1.0$ in
this limit.

**However**, trained networks organise their weights into structured patterns:
attention heads form near-orthonormal pairs, MLP neurons often have complementary
partners, and weight matrices develop block structure. The Q² run-reduction step
applied to weight rows (§D-3.1) can be used diagnostically to measure this
structure: a low transition density (many consecutive identical symbols) implies
longer runs and higher compressibility.

The empirical prediction is that Q²-QAT weights will compress to $r \approx 1.1$–$1.3$
under zstd-22 — more than a random 2-bit stream but less than the int5/int6 models
(which have float-shaped distributions amenable to entropy coding).

---

## 4 Architecture: Liquid Time Constant Networks

### 4.1 The parameter inefficiency of attention

Standard transformer attention has quadratic time complexity $O(n^2 d)$ in sequence
length and requires four weight matrices of size $d \times d$ per head per layer.
For a model with hidden dimension $d$ and $L$ layers:

$$N_{\text{attn}} = 4 L d^2$$

In the Parameter Golf setting, attention is expensive: each attention layer in a
512-dim model costs $4 \times 512^2 = 1.05 \text{ M}$ parameters, and the
information content is dominated by the key-value store, not the query-key
interaction.

For short-context tasks (1024–2048 tokens, as used in current winning entries), the
attention mechanism is also overqualified: most of the model's context budget is
already consumed by the first $\sim$10 positions, and positions beyond that
contribute diminishing marginal information.

### 4.2 Closed-form Continuous-time (CfC) layers

Hasani et al.'s **Closed-form Continuous-time** (CfC) networks provide a
parameter-efficient alternative. The CfC layer solves the Liquid Time Constant
(LTC) ODE:

$$\dot{h}(t) = -\left[\frac{1}{\tau} + f(h(t), x(t); \theta)\right] h(t) + f(h(t), x(t); \theta)$$

analytically, yielding a closed-form update:

$$h(t + \Delta t) = \exp\!\left(-A_1(t) \cdot \Delta t\right) \odot h(t) + \frac{A_2(t)}{A_1(t)} \cdot \left[1 - \exp\!\left(-A_1(t) \cdot \Delta t\right)\right]$$

where $A_1, A_2$ are functions of the input $x(t)$ and current state $h(t)$, and
$\exp$ denotes the elementwise exponential. This closed form:

1. Eliminates the numerical integration loop of vanilla LTC networks.
2. Provides causal, single-pass inference: each token updates the state $h$ in
   $O(d)$ time, independent of sequence length.
3. Requires only two linear projections ($A_1, A_2$) plus the state update — far
   fewer parameters than a full attention block.

**Parameter count comparison.** For hidden dimension $d$:

| Block type | Parameters per layer |
|:-----------|:--------------------:|
| Full MHA | $4d^2$ |
| GQA (4 KV heads) | $\approx 3.5 d^2$ |
| CfC (closed-form) | $\approx 2 d^2 + 2d$ |
| CfC (compact) | $\approx d^2 + 2d$ |

The CfC layer requires approximately $d^2$ fewer parameters per layer than
full attention. Over $L$ layers, this frees:

$$\Delta N = L \cdot d^2 \text{ parameters}$$

For $L = 10$, $d = 512$: $\Delta N = 10 \times 512^2 = 2.6 \text{ M}$ parameters
freed for other components (larger MLP, larger BigramHash table, or more layers).

### 4.3 Liquid Foundation Models (LFM 2.5) as a template

Liquid AI's **LFM 2.5** model demonstrates the viability of hybrid recurrent +
attention architectures at production scale. The LFM 2.5 architecture uses:

- **10 LIV (Liquid Integrated Vision/Language) Convolution Blocks:** CfC-based
  sequential processors that provide O(1) per-token memory through recurrent state.
- **6 GQA (Grouped Query Attention) Blocks:** Standard attention for positional
  cross-token mixing.
- **32k token trained context:** Achievable because LIV blocks handle most of the
  context without O(n²) cost.

The LFM 2.5 result demonstrates that attention is not required for most of the
model's depth — the CfC state provides sufficient long-range memory. Attention
is used selectively for in-context reasoning and positional disambiguation.

For the Parameter Golf setting, the 32k context is not needed. But the principle
transfers: **replace most attention layers with CfC, keep a few GQA layers for
in-context mixing.**

### 4.4 CfC layers and Q²-QAT synergy

The Q² structural quantization (§D-2.4) is particularly well-suited to CfC weights
for two reasons:

1. **State update weights have complement structure.** The two matrices $A_1$ and
   $A_2$ in the CfC update equation have a natural complement relationship: one
   controls the decay rate and the other controls the input integration rate.
   The Q² complement involution $\theta(A) = C$, $\theta(B) = D$ (§D-2.8) encodes
   this opposition directly — strong-decay and strong-integration are complements
   in the same way that strong-negative and strong-positive activations are.

2. **Fewer weights need high precision.** CfC state updates involve sigmoid
   activations, which saturate at $\pm 1$. Near the saturation region, the exact
   weight value matters less than its sign and magnitude class — precisely what Q²
   preserves (§D-1.5). The two cells $A$ (strong negative, below $-\tau^{\ast}$)
   and $D$ (strong positive, above $+\tau^{\ast}$) correspond to the saturation
   regime; $B$ and $C$ correspond to the linear-response regime near zero.

### 4.5 Geode-derived layer layout

LFM 2.5's 10:6 CfC:GQA ratio was found empirically. Note that 10:6 cannot be
reduced to 5:3: the numbers are absolute layer counts (10 CfC + 6 GQA = 16 layers
total), not a bare ratio. Reducing to 5:3 would describe a different 8-layer
model, halving the depth. The Geode factorization (§D-4.1) provides a principled
derivation that eliminates the guesswork.

The generating function for Q²'s transition sequences:

$$S(x) - 1 = \frac{4x}{1-3x} = \underbrace{4x}_{S_1} \cdot \underbrace{\frac{1}{1-3x}}_{G}$$

decomposes into two factors with a direct architectural interpretation:

- **$S_1 = 4x$**: the first symbol has **4 choices** — the 4 coarse quantization
  cells. Architecturally: **4 GQA blocks**, each establishing the broadest
  context structure (equivalent to selecting one of 4 block files in the
  transition key, §D-3.4).

- **$G = 1/(1-3x) = 1 + 3x + 9x^2 + \cdots$**: each subsequent symbol has
  **3 choices** — refinement within the established coarse cell.
  Architecturally: **3 CfC blocks per GQA block**, each performing one 3-way
  refinement step within the coarse context.

This gives the layer pattern:

$$\underbrace{[\text{GQA},\ \text{CfC},\ \text{CfC},\ \text{CfC}]}_{\text{one Geode level}} \times 4 = 16 \text{ layers total}$$

**4 GQA + 12 CfC**, with CfC:GQA ratio **3:1** — compared to LFM 2.5's empirical
10:6 = 1.67:1. The Geode predicts a more CfC-heavy architecture, consistent with
the hypothesis that less attention is needed at the short-context (2048-token)
parameter-golf scale.

**Information accumulated at each stage.** The Geode gives the bits of
structural information captured at depth $k$:

- After 1 GQA block: $\log_2 4 = 2$ bits of coarse context.
- After each additional CfC step: $+\log_2 3 \approx 1.585$ bits of refinement.
- After all 16 layers (4 coarse + 12 refinement): $4 \times (2 + 3 \times \log_2 3) \approx 27.0$ bits.

This sits within the 51.1-bit capacity of the full 32-symbol key (§D-3.6),
confirming the 16-layer model can represent sufficient structural information for
2048-token language modeling.

**Layer position mapping:**

| Layer | Type | Geode node | Purpose |
|:-----:|:-----|:----------:|:--------|
| 1 | GQA | $S_1$ root | Coarse context — 4 choices ($r_0$, §D-3.2) |
| 2–4 | CfC × 3 | $G$ level 1 | First refinement — 3 choices per step |
| 5 | GQA | $S_1$ reset | Re-establishes coarse context |
| 6–8 | CfC × 3 | $G$ level 2 | Second refinement |
| 9 | GQA | $S_1$ reset | Re-establishes coarse context |
| 10–12 | CfC × 3 | $G$ level 3 | Third refinement |
| 13 | GQA | $S_1$ reset | Final coarse context |
| 14–16 | CfC × 3 | $G$ level 4 | Fourth refinement |

The GQA layers act as "semantic resets" — attending across the full token
sequence to re-establish coarse structure; the CfC layers refine within that
structure token-by-token using recurrent state.

---

## 5 The Combined Strategy

### 5.1 Architecture

The proposed architecture for the Parameter Golf submission is a **Q²-QAT hybrid
LTC-Transformer**, combining:

1. **Q² 2-bit QAT** for all weight matrices (attention, MLP, CfC state).
2. **Hybrid depth:** Geode-derived layout (§4.5) — [GQA, CfC, CfC, CfC] × 4
   = 16 layers (4 GQA + 12 CfC).
3. **BigramHash** vocabulary embedding: a hash table of bigram statistics stored
   as part of the 16 MB artifact.
4. **Sliding window evaluation** at stride 64.

```mermaid
flowchart TD
    subgraph Model["Q2-QAT Hybrid LTC-Transformer (Geode layout)"]
        direction TB
        emb["Token Embedding\n(FP16, tied)"]
        bh["BigramHash\n(bigram log-probs, 2-4 MB)"]
        subgraph Stack["16-layer Geode stack: (GQA, CfC, CfC, CfC) x4"]
            direction TB
            gqa1["GQA Block x4\n(Q2 2-bit, coarse: 4 choices)"]
            cfc1["CfC Block x12\n(Q2 2-bit, refine: 3 choices each)"]
        end
        lm_head["LM Head\n(tied to embedding)"]
    end
    emb --> Stack
    bh -->|"log-prob prior"| lm_head
    Stack --> lm_head
```

**Hidden dimension and layer count.** With 64 M parameters at 2 bits per weight,
packed 4 per byte, and BigramHash(10240) consuming ~4 MB:

$$N_{\text{model}} \approx \frac{(16 \times 10^6 - 4 \times 10^6 - 50{,}000) \times 4 \times 8}{8} \approx 48 \text{ M effective parameters}$$

At hidden dimension $d = 768$ with $n_{\text{kv}} = 4$ KV heads and MLP ratio 3×,
the parameter count breaks down by component:

- **4 GQA blocks:** Q ($d^2$) + K ($d^2/3$) + V ($d^2/3$) + O ($d^2$) +
  MLP-up/gate/down (3 × 3$d^2$) = $(8/3 + 9)d^2 \approx 11.67d^2$ each.
- **12 CfC blocks:** $A_1$ ($2d^2$) + $A_2$ ($2d^2$) + out ($d^2$) = $5d^2$ each.

$$N \approx 4 \times 11.67 d^2 + 12 \times 5 d^2 = 106.7 d^2 \approx 63 \text{ M at } d = 768$$

This matches the 64 M capacity projected in §3.1. Tuning $d$ to 700–730 leaves
room for the BigramHash table; $d = 768$ fills the budget tightly without it.

### 5.2 Quantization scheme

All linear weight matrices $W \in \mathbb{R}^{m \times n}$ are quantized to Q²
symbols $\{A, B, C, D\} = \{0, 1, 2, 3\} \subset \mathbb{Z}_4$. The quantization
threshold applied during training:

$$\tau^{\ast} = \frac{\Phi^{-1}(3/4)}{\sqrt{n}} \approx \frac{0.6745}{\sqrt{n}}$$

is computed from the current batch statistics (the empirical 25th and 75th
percentile of each row) and updated every 1024 training steps — the same
reservoir-calibration strategy described in §D-2.5 for activation quantization.

The straight-through estimator (STE) propagates gradients through the
quantization step:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}} \approx \frac{\partial \mathcal{L}}{\partial \hat{W}_{ij}} \cdot \mathbf{1}\!\left[|W_{ij}| \leq \kappa\right]$$

where the passthrough window $\kappa$ is set to exclude extreme outliers that
would otherwise receive large gradients through the saturating threshold.

**Packed storage.** Q² symbols are Gray-encoded (§D-2.7) and packed 4 per byte
using the same packing scheme as the WebAssembly kernel in `src/q2.wat`:

```
byte = (g[4i] << 6) | (g[4i+1] << 4) | (g[4i+2] << 2) | g[4i+3]
```

This layout is identical to the activation quantization in `src/q2.wat`, making
the q2.ts library directly usable for weight packing at checkpoint export time.

### 5.3 Mixed-precision allocation

Not all weight matrices benefit equally from 2-bit precision. Following the
Geode mixed-precision framework (§D-4.3) and the empirical finding of QuES
(§R-2.4) that arithmetic-reasoning channels require higher precision:

- **Embedding layer:** Tied FP16. The embedding matrix is not quantized; it
  serves as the interface between the discrete token space and the continuous
  weight space. FP16 embeddings with 10240 vocabulary and 768 dimensions cost
  $10240 \times 768 \times 2 \approx 15.7$ MB — too large. With vocabulary 1024:
  $1024 \times 768 \times 2 = 1.57$ MB, acceptable.
- **Q² 2-bit for all linear layers:** All attention projections, CfC state
  matrices, and MLP weight matrices are quantized to Q² 2-bit.
- **Layer norm parameters:** Kept in FP16 (negligible count, critical for
  training stability).
- **BigramHash:** Stored as FP16 log-probabilities, taking 4–8 MB of the budget.

### 5.4 Training strategy

The training recipe follows the current SOTA structure with Q²-specific additions:

| Component | Setting | Rationale |
|:----------|:--------|:----------|
| Optimizer | Muon (Nesterov + spectral normalisation) | Current SOTA |
| Weight decay | 0.04 | Current SOTA |
| Learning rate schedule | cosine with warmup 200 steps | Standard |
| SWA (stochastic weight averaging) | last 40% of training | Current SOTA |
| Q² threshold update | every 1024 steps, reservoir size 1024 | §D-2.5 |
| STE passthrough | $\kappa = 3\tau^{\ast}$ | Standard QAT practice |
| Gradient clipping | 1.0 | Training stability |
| Sequence length | 2048 | Context for language modeling |
| Evaluation | sliding window stride 64 | Current SOTA |
| Vocabulary | SP-1024 (SentencePiece, 1024 tokens) | Matches challenge baseline |

**Warm-up from FP32 pre-training.** A common failure mode of QAT is that the
model begins training with random 2-bit weights that are too noisy for the
complement structure to emerge. The recommended warm-up strategy:

1. Train for 500 steps in FP32 with standard initialisation (OrthoInit for
   attention, standard Kaiming for MLP).
2. Apply Q² quantization to the FP32 checkpoint with empirical threshold
   calibration.
3. Continue training with Q²-QAT from the quantized checkpoint.

This mirrors the BitNet finding (§R-3.1) that training-from-scratch QAT requires
a brief float-precision warm-up to establish the initial activation distribution
before the quantization constraint is imposed.

### 5.5 LIV cache-line packing and byte tokenization

Two additional techniques, compatible with the Geode architecture, that can
improve parameter efficiency and reduce artifact size further:

#### 5.5.1 LIV cache-line packing

LIV (Liquid Integrated Vision/Language) symbols use 5-bit quantisation (int5,
32 levels). A 64-bit register holds:

$$12 \times 5 + 2 + 2 = 64 \text{ bits}$$

That is, **12 LIV symbols** (60 bits) plus a **2-bit Q² tag** and 2 unused bits.
The Q² tag is a coarse-context label — one of 4 values matching the
$S_1 = 4x$ coarse level of the Geode factorization — that identifies which
GQA "bucket" produced the 12-symbol LIV block.

**Packing layout** (bits 63 → 0, MSB-first):

```
[sym0(5)] [sym1(5)] … [sym11(5)] [tag(2)] [00]
 bit 63                 bits 8:4   bits 3:2  1:0
```

sym0 → bits [63:59], sym1 → bits [58:54], …, sym11 → bits [8:4]; tag → bits
[3:2]; bits [1:0] are unused (zero).

This layout has two computable advantages:

1. **Parallel dispatch by tag.** The 2-bit tag [0..3] partitions the packed
   words into 4 groups. Each GPU streaming multiprocessor processes one tag
   group, maximizing cache locality and SM utilization without coordination
   overhead.

2. **The 10-LIV codon representation.** Taking only the top 10 × 5 = 50 bits,
   the block can be interpreted as **two 5 × 5 binary matrices** $M_1$ and
   $M_2$ (25 + 25 = 50 bits). Their Boolean matrix product:

   $$C_{ij} = \bigvee_k \left[(M_1)_{ik} \wedge (M_2)_{kj}\right]$$

   is a deterministic function of the pair. This means:
   - A "codon" (the Boolean product $C$) uniquely identifies the (M₁, M₂)
     pair up to equivalence.
   - Any candidate pair can be verified against a stored codon in $O(25)$
     Boolean operations — cheap on GPU via warp-level bitwise ops.
   - The remaining 14 bits (2 LIV sym11 bits + 2-bit tag + 2 unused) serve as
     a sequence index ordering codons for distributed processing.

   This convolution-verifiable structure mirrors the role of the Q² transition
   key (§D-3.3) but at a coarser 5-bit resolution, providing a hardware-level
   checksum for the LIV block without extra storage.

`scripts/q2_pack.py` exports `pack_liv_cacheline` and `unpack_liv_cacheline`
that implement this layout on GPU-resident tensors.

#### 5.5.2 Byte tokenization — skip the tokeniser encoder

The SP-1024 tokenizer introduces a pre-processing step (encode/decode) that
costs latency and requires a vocabulary embedding matrix of size
$V \times d = 1024 \times 768 \approx 1.6$ MB.

At the byte level, vocabulary is always exactly 256, regardless of corpus
language or domain:

| Tokenization | Vocab | Embedding cost | Tokenizer | Compression |
|:-------------|:-----:|:--------------:|:---------:|:-----------:|
| SP-1024 | 1024 | 1.57 MB | Required | ~3× sub-word |
| Raw bytes | 256 | 0.39 MB | None | 1× byte |

The embedding savings alone free ~1.2 MB — enough for additional model
parameters at 2 bits/weight ($\approx 5$ M extra weights).

**Training on raw bytes.** Set `BYTE_TOKENS=1` to enable byte mode in
`scripts/train_q2_ltc.py`. The data shards are read as raw `uint8` streams;
each byte becomes a token id in [0, 255]. No SentencePiece encode/decode step
is needed anywhere in the pipeline:

```bash
BYTE_TOKENS=1 VOCAB_SIZE=256 torchrun --standalone --nproc_per_node=8 \
    scripts/train_q2_ltc.py
```

The model sees the same FineWeb text; the challenge scorer operates on bytes
and computes bpb directly on the byte sequence, so there is no evaluation
penalty for skipping the tokenizer.

---

## 6 Implementation Roadmap

The implementation is in two Python scripts in `scripts/`:

- **`scripts/q2_pack.py`** — GPU-accelerated Q² weight packing and unpacking.
- **`scripts/train_q2_ltc.py`** — Complete training script: Q²-QAT, Geode
  architecture, Muon optimizer, SWA, and artifact packaging.

### 6.1 Phase 1 — Q² weight packing (`scripts/q2_pack.py`)

`q2_pack.py` converts a PyTorch state dict to the Q2BN binary format and back.
All quantisation operations run on GPU when available, falling back to CPU.

Key functions:

- `empirical_tau(W)` — per-row 75th-percentile threshold (§D-2.5), vectorised
  on GPU via `torch.quantile`.
- `q2_quantise(W, tau)` — four-cell quantisation to {A=0,B=1,C=2,D=3} using
  three vectorised comparisons with no Python loops.
- `gray_encode(sym)` / `gray_decode(gray)` — Gray map φ: sym XOR (sym >> 1).
- `pack_symbols(gray)` / `unpack_symbols(packed, n)` — 4 symbols per byte,
  MSB-first; packing uses a single batched `|` operation over the 4-symbol groups.
- `pack_state_dict(state_dict, out_path)` — serialise to Q2BN format.
- `unpack_state_dict(in_path, device)` — deserialise back to float tensors.
- `pack_liv_cacheline(symbols, seq_tags)` / `unpack_liv_cacheline(packed, n)` —
  LIV 5-bit cache-line packing (§5.5.1): 12 LIV + 4-bit Q² tag per 64-bit word.

CLI usage:

```bash
# Pack a PyTorch checkpoint to Q2 binary:
python scripts/q2_pack.py model.pt model.q2bin

# Inspect a packed file:
python scripts/q2_pack.py --unpack model.q2bin
```

### 6.2 Phase 2 — Training script (`scripts/train_q2_ltc.py`)

`train_q2_ltc.py` is the complete training script. It implements:

- **`Q2Linear`** — `nn.Linear` subclass with STE quantisation.  Behaves as a
  standard linear layer during FP32 warm-up; call `activate_q2()` to switch.
  Refreshes τ* every `tau_update_every` steps from the empirical weight
  distribution.

- **`CfCBlock`** — One Geode G-level (3-way refinement).  Runs the closed-form
  LTC update per token; state `h` propagates across the sequence with no KV
  cache.  All projections are `Q2Linear`.

- **`GQABlock`** — One Geode S1-level (4-way coarse selection).  Uses
  `F.scaled_dot_product_attention` (FlashAttention kernel on H100) with GQA
  head sharing.  SwiGLU MLP with 3× expansion.  All projections are `Q2Linear`.

- **`Q2LTCModel`** — Full 16-layer model with Geode layout
  `[GQA, CfC, CfC, CfC] × 4`.  OrthoInit weights; tied embeddings and LM head.

- **`Muon`** — Nesterov momentum + per-matrix spectral normalisation.  Prevents
  large weight moves from disrupting Q² complement structure during QAT.

- **Training loop** — `torch.compile(mode="max-autotune")` for kernel fusion;
  bfloat16 autocast; gradient accumulation; cosine LR + warmup; SWA from 60% of
  training; sliding-window validation; automatic Q2BN + zstd-22 packaging.
  Byte-mode training (`BYTE_TOKENS=1`) skips the tokeniser encoder entirely
  (§5.5.2).

Single-GPU smoke test:

```bash
MAX_STEPS=200 BATCH_TOKENS=8192 python scripts/train_q2_ltc.py
```

Full 8×H100 run (SP-1024 tokens):

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_q2_ltc.py
```

Full 8×H100 run (raw bytes, no tokeniser):

```bash
BYTE_TOKENS=1 torchrun --standalone --nproc_per_node=8 scripts/train_q2_ltc.py
```

### 6.3 Phase 3 — Artifact packaging (built into training script)

At the end of training, `train_q2_ltc.py` automatically:

1. Selects the SWA-averaged model (or the final model if SWA has not started).
2. Packs all weight matrices to Q2BN via `q2_pack.pack_state_dict`.
3. Compresses with zstd level 22 (requires `pip install zstandard`).
4. Reports the total artifact size and flags if it exceeds 16 MB.

To trigger packaging on an existing checkpoint:

```bash
python -c "
import torch, sys
sys.path.insert(0, 'scripts')
import q2_pack
sd = torch.load('checkpoint.pt', map_location='cpu', weights_only=True)
n = q2_pack.pack_state_dict(sd.get('model', sd), 'model.q2bin')
print(f'{n/1e6:.3f} MB')
"
```

---

## 7 Performance Projections

### 7.1 Parameter capacity

| Method | Bits/weight | Parameters in 16 MB | Relative capacity |
|:-------|:-----------:|:-------------------:|:-----------------:|
| Naive baseline (int8) | 8 | ~11 M | 1.0× |
| Current SOTA (int5/int6) | 5.5 | ~23 M | 2.1× |
| Q² 2-bit | 2.0 | ~64 M | 5.8× |
| Q² 2-bit + zstd compression | ~1.7 | ~75 M | 6.8× |

### 7.2 Scaling law projection

Under the Chinchilla scaling law, language model loss scales as:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

with $E \approx 1.61$ nats/token (irreducible entropy), $\alpha \approx 0.34$,
$\beta \approx 0.28$.

In the Parameter Golf setting $D$ is effectively unlimited (8B tokens available);
the bottleneck is $N$. Moving from 23 M to 64 M parameters at the same data
volume predicts:

$$\Delta L \approx A \cdot \left(N_{23M}^{-\alpha} - N_{64M}^{-\alpha}\right) \approx A \cdot (23M^{-0.34} - 64M^{-0.34})$$

For a rough estimate with $A \approx 406.4$ (Chinchilla value):

$$\Delta L \approx 406.4 \times (4.09 \times 10^{-3} - 2.71 \times 10^{-3}) \approx 0.056 \text{ nats/token}$$

Converting to bpb: $\Delta \text{bpb} = \Delta L / \ln 2 \approx 0.081$.

This suggests a projected bpb of $1.1428 - 0.081 \approx 1.06$ for the pure
scaling benefit of 2.8× more parameters — ignoring any additional benefit from
the CfC architecture's superior parameter efficiency per layer.

**Caveat.** This projection assumes that 2-bit Q² model quality matches 5-bit
quality at the same parameter count, which requires successful QAT. The
BitNet b1.58 (§R-3.1) and binary/ternary weight literature (§R-3.2) consistently
show that QAT-from-scratch at ≥1.58 bits is competitive with post-training
quantization at 4–5 bits. The 2-bit Q² point is between ternary (1.58 bits) and
binary-weighted quantization (1 bit), and the complement structure of
$\mathbb{Z}_4$ provides richer inductive bias than either.

### 7.3 The CfC efficiency multiplier

The CfC parameter efficiency argument is harder to quantify analytically. The LFM
2.5 result (matching or exceeding GPT-class models on language benchmarks with
far fewer attention operations) suggests that the CfC recurrent state provides
$O(d)$ effective context memory at $O(d^2)$ parameter cost — the same
asymptotic complexity as attention, but with lower constant factors because:

- No key-value cache growth with sequence length.
- No positional encoding overhead.
- State update is a sigmoid multiply-add, not a softmax over all prior keys.

For the 10-minute training constraint on 8×H100, the CfC blocks train faster per
step than attention blocks of equal parameter count because there is no CUDA
FlashAttention kernel overhead for the CfC state update (a simple element-wise
operation).

### 7.4 Summary projection

| Component | Estimated bpb improvement |
|:----------|:-------------------------:|
| Current SOTA baseline | 1.1428 |
| Q² 2-bit QAT (parameter scaling alone) | -0.08 |
| CfC architecture (parameter efficiency) | -0.02 to -0.05 (estimated) |
| Larger BigramHash enabled by space saving | -0.01 to -0.02 |
| **Projected total** | **~1.00 to 1.03** |

A score of 1.00–1.05 bpb would represent a substantial improvement over the
current SOTA (1.1428 bpb) — an advance of roughly 0.08–0.14 bpb, well above the
0.005-nat (~0.007 bpb) significance threshold required for leaderboard submission.

### 7.5 Williams SpaceTime bound and optimal bit width

**Ryan Williams (2025)** proved that any multitape Turing machine running in time
$t(n)$ can be simulated in space:

$$S = \mathcal{O}\!\left(\sqrt{t(n) \cdot \log t(n)}\right)$$

(Williams, *Simulating Time With Square-Root Space*, STOC 2025 / arXiv:2502.17779.)
This is a dramatic improvement over the Hopcroft–Paul–Valiant 1975 bound of
$\mathcal{O}(t / \log t)$, and it gives a rigorous information-theoretic relationship
between computation time and storage space.

#### Applying Williams to the 16 MB / 10-minute constraint

**Available computation** (8×H100, BF16, 10 min):

$$t = 8 \times 989 \times 10^{12} \times 600 \approx 4.75 \times 10^{18} \text{ FLOPS}$$

**Williams lower bound on space** needed to faithfully simulate $t$:

$$S_{\min} = \mathcal{O}\!\left(\sqrt{4.75 \times 10^{18} \times \log_2(4.75 \times 10^{18})}\right)
           = \mathcal{O}\!\left(\sqrt{4.75 \times 10^{18} \times 62}\right)
           \approx 1.72 \times 10^{10} \text{ bits} \approx 2.15 \text{ GB}$$

**Our artifact space**: $S = 16 \times 10^6 \times 8 = 1.28 \times 10^8$ bits (16 MB).

$$\frac{S}{S_{\min}} \approx \frac{1.28 \times 10^8}{1.72 \times 10^{10}} = 0.0075 = 0.75\%$$

We have **0.75% of the Williams-implied storage**. This places the challenge firmly
in the deep-compression regime: the model is far too small to faithfully represent
all computation in the training run. Only the most structured, compressible patterns
in FineWeb can be captured.

#### Reverse: what does 16 MB imply about effective computation?

Inverting $S^2 \approx t \cdot \log_2 t$ for $S = 1.28 \times 10^8$ bits:

$$S^2 = 1.638 \times 10^{16} \implies t_{\max} \approx 3.4 \times 10^{14} \text{ FLOPS}$$

**Interpretation**: A 16 MB model can faithfully encode the structure of approximately
$3.4 \times 10^{14}$ FLOPS of computation — or about $7 \times 10^{-3}$% of the
10-minute H100 training budget. The remaining training FLOPS refine the model's
weights without encoding qualitatively new information (they push the stored structure
toward the FineWeb distribution, but cannot expand the model's capacity).

This is why the challenge rewards **compression per bit above all else**: every bit
is precious. Any format that wastes bits on alignment padding, metadata overhead, or
suboptimal bit-width penalizes the final score.

#### Cache-line efficiency by bit width

A 64-byte cache line holds 512 bits. The waste per line and total parameter budget
for each integer bit width. The table shows **GPU-native 64-bit register alignment**
(CUDA operates on 64-bit or 32-bit aligned chunks):

| Bits/weight | Params/register | Wasted bits/register | Params/cache-line | Effective N (16 MB) |
|:-----------:|:---------------:|:--------------------:|:-----------------:|:-------------------:|
| 1 | 64 | 0 | 512 | 128 M |
| **2 (Z₄)** | **32** | **0** | **256** | **64 M** |
| 4 (Z₈) | 16 | 0 | 128 | 32 M |
| **5 (int5)** | 12 | **4** | **96** | **~24 M** |
| **6 (int6)** | 10 | **4** | **80** | **~20 M** |
| 8 (Z₁₆) | 8 | 0 | 64 | 16 M |

Power-of-2 bit widths (1, 2, 4, 8) divide evenly into 64-bit registers — **zero
waste**. For int5 and int6, packing per 64-bit register leaves 4 unused bits
(6.25% per register). Across 2,000,000 registers in 16 MB:

$$2{,}000{,}000 \times 4 \text{ bits} = 8{,}000{,}000 \text{ bits} = 1 \text{ MB wasted}$$

That 1 MB recovers $\approx 4$ M additional Z₄ parameters (1 MB × 8 / 2 bits =
4 M params) — enough to noticeably move bpb via Chinchilla scaling (§7.2).

#### The LIV bit-width question resolved

The current SOTA uses post-training quantization to **int5** (LFM 2.5 GGUF format).
Several parallel analyses have been debating whether LIV blocks need 4 or 5 bits.
The Williams + cache-line analysis gives a definitive answer:

1. **For Q²-QAT training from scratch**: use Z₄ **2-bit** throughout.  
   This maximises $N = 64$ M parameters — the information-theoretically optimal
   choice for integer bit widths, given that 2-bit is the minimum meaningful
   representation (1-bit binary weights are viable but lose the complement structure
   of $\mathbb{Z}_4$ that makes Q² quantization uniquely effective).

2. **For LIV-format post-training compression**: **4-bit (Z₈)** strictly dominates
   **5-bit (int5)** for GPU-aligned storage because 4-bit has zero register waste
   ($N = 32$ M) while int5 wastes 4 bits per register ($N \approx 24$ M effective,
   not 25.6 M nominal).

3. **The §5.5.1 scheme** (12 LIV × 5-bit + 4-bit Q² tag = 64 bits exactly) IS a
   perfectly aligned 64-bit word — no register waste — but allocates 4 of 64 bits
   to metadata rather than weight storage, giving an effective density of
   $64/12 = 5.33$ bits/LIV. This is useful for parallel dispatch and codon
   verification, but less dense than pure Z₄ (2 bits/param) or Z₈ (4 bits/param).

**Bottom line**: Our Q²-QAT approach uses Z₄ 2-bit weights for all model parameters.
This is the unique integer bit-width that simultaneously:
- Achieves maximum $N = 64$ M parameters in the 16 MB budget
- Packs perfectly into 64-bit registers and 64-byte cache lines (zero waste)
- Preserves the $\mathbb{Z}_4$ complement structure and Lee metric
- Falls within the training-from-scratch QAT regime proven competitive by BitNet (§R-3.1)

The int5/int6 debate applies to post-training quantization of float-trained models.
For QAT-from-scratch, 2-bit is the correct choice from both a Williams perspective
(maximise $N$) and an algebraic one (preserve $\mathbb{Z}_4$ ring structure).

#### Reconciliation with parallel analyses

Two parallel analyses (in `PARAMETER_GOLF_REVISED.md` and `docs/parameter-golf.md`
on the `main` branch) reach compatible conclusions:

- `PARAMETER_GOLF_REVISED.md` correctly identifies that **odd bit-widths are
  suboptimal for cache alignment** and recommends power-of-2 widths. Williams
  confirms this: every wasted bit reduces $N$, directly increasing bpb.

- `docs/parameter-golf.md` recommends mixed int5/int6 precision, which is the
  leaderboard SOTA approach. The Williams analysis shows this is suboptimal vs.
  2-bit QAT because it achieves $N_{\text{eff}} \approx 24$ M at int5 (not the
  nominal 25.6 M, due to register alignment), while Q² 2-bit achieves $N = 64$ M.
  From §7.2, the predicted $\Delta\text{bpb} \approx 0.08$ from this parameter
  gap alone.

The three analyses converge on: **maximum parameters at lowest possible bit-width
with perfect cache alignment** — which is Q² 2-bit.

---

## 8 References

- OpenAI Parameter Golf challenge. <https://openai.com/index/parameter-golf/>
- OpenAI Parameter Golf GitHub repository. <https://github.com/openai/parameter-golf>
- Williams, R. (2025). Simulating Time With Square-Root Space. *Proc. STOC 2025*.
  arXiv:2502.17779. (§7.5)
- Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosse-Wentrup, M. (2021). Liquid
  Time-constant Networks. *AAAI-2021*. arXiv:2006.04439.
- Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M.,
  Teschl, G., & Rus, D. (2022). Closed-form Continuous-time Neural Networks.
  *Nature Machine Intelligence* 4, 992–1003. arXiv:2106.13898.
- Liquid AI. LFM 2.5 Technical Report. (2025).
  <https://www.liquid.ai/research/lfm-2-5>
- Ma, S. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in
  1.58 Bits. arXiv:2402.12263. (§R-3.1)
- Wildberger, N. J. & Rubine, D. (2025). A Hyper-Catalan Series Solution to
  Polynomial Equations, and the Geode. *Amer. Math. Monthly* 132:5, 383–402.
  (§D-4.1)
- Hammons, A. R., Kumar, P. V., Calderbank, A. R., Sloane, N. J. A., & Solé, P.
  (1994). The $\mathbb{Z}_4$-linearity of Kerdock, Preparata, Goethals, and related
  codes. *IEEE Trans. Inform. Theory* 40:2, 301–319. (§D-2.7)
- NanoGPT Speedrunning. <https://github.com/KellerJordan/modded-nanogpt>
