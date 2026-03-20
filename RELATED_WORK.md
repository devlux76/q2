# Quaternary Quantization: Related Work

> **Related documents:** [DESIGN.md](DESIGN.md) · [PREDICTIONS.md](PREDICTIONS.md)
>
> This document surveys existing work on quaternary (2-bit, 4-level) quantization in
> machine learning. It compares, contrasts, and distinguishes those approaches from
> the structural quantization scheme described in [DESIGN.md](DESIGN.md), and
> identifies findings from the literature that are directly relevant to Q2.

---

## Contents

1. [The Fundamental Distinction](#1-the-fundamental-distinction)
2. [Reconstruction-Based 2-bit Methods](#2-reconstruction-based-2-bit-methods)
   - 2.1 [GPTQ and AWQ — the baseline](#21-gptq-and-awq--the-baseline)
   - 2.2 [BQQ — Binary Quadratic Quantization (NeurIPS 2025)](#22-bqq--binary-quadratic-quantization-neurips-2025)
   - 2.3 [QUAD — Quantization and Parameter-Efficient Tuning for LLMs](#23-quad--quantization-and-parameter-efficient-tuning-for-llms)
   - 2.4 [QuES — Quantized Expert Scaling](#24-ques--quantized-expert-scaling)
   - 2.5 [NVFP4 — 4-bit NV Float](#25-nvfp4--4-bit-nv-float)
3. [Alternative Low-Precision Schemes](#3-alternative-low-precision-schemes)
   - 3.1 [BitNet — 1.58-bit ternary architectures](#31-bitnet--158-bit-ternary-architectures)
   - 3.2 [Binary and ternary weight networks](#32-binary-and-ternary-weight-networks)
4. [Domain-Specific Applications](#4-domain-specific-applications)
   - 4.1 [Edge and embedded hardware](#41-edge-and-embedded-hardware)
   - 4.2 [RNNs and sentiment analysis](#42-rnns-and-sentiment-analysis)
   - 4.3 [CNN accelerators for industrial monitoring](#43-cnn-accelerators-for-industrial-monitoring)
   - 4.4 [Quaternary neural belief propagation (BP4) for QLDPC codes](#44-quaternary-neural-belief-propagation-bp4-for-qldpc-codes)
   - 4.5 [Image steganography (OPMS-QQGE)](#45-image-steganography-opms-qqge)
5. [Accuracy vs. Efficiency Trade-offs Across the Literature](#5-accuracy-vs-efficiency-trade-offs-across-the-literature)
6. [Key Distinctions: Q2 vs. the Field](#6-key-distinctions-q2-vs-the-field)
7. [Borrowed Insights](#7-borrowed-insights)
8. [References](#references)

---

## 1 The Fundamental Distinction

All work surveyed here uses the quaternary (4-level) alphabet in some form. The
central distinction — from which all others follow — is between **reconstruction
quantization** and **structural quantization**.

**Reconstruction quantization** (the dominant paradigm in the literature) minimises
pointwise reconstruction error. Given a weight matrix $W$, the objective is:

$$\min_{\hat{W}} \| W - \hat{W} \|_F^2$$

subject to $\hat{W}$ having entries drawn from a small codebook. The metric is the
Frobenius norm. The quantity being preserved is the weight value itself, so that
the quantised model behaves as close to the full-precision model as possible.

**Structural quantization** (the Q2 approach, described in §D-2.4) has a different
objective: preserve relational and topological structure — distances, trajectories,
and complement relationships — rather than pointwise values. The metric is the Lee
distance on $\mathbb{Z}_4$. The quantity being preserved is the geometry of the
activation space, not the magnitude of individual activations.

These two objectives are compatible at the level of alphabet size (both use 4 levels)
but orthogonal in what they optimize. A reconstruction quantizer can be evaluated
on perplexity, zero-shot accuracy, and task benchmarks. A structural quantizer is
evaluated on retrieval fidelity, distance preservation, and the downstream quality
of the transition key.

The distinction matters because insights transfer only within objective class. Most
of the literature surveyed below optimises for reconstruction; Q2 optimises for
structure. Some insights nonetheless transfer; Section 7 identifies which ones.

---

## 2 Reconstruction-Based 2-bit Methods

### 2.1 GPTQ and AWQ — the baseline

GPTQ (Frantar et al. 2022) applies a layer-wise second-order weight update to
compensate for quantization error. For each layer, it quantises weights one column
at a time and updates the remaining unquantised weights to absorb the error, using
the inverse Hessian of the layer's input distribution:

$$\delta W_{\text{remaining}} = -q_{\text{error}} \cdot H^{-1}$$

AWQ (Lin et al. 2023) takes a different route: it searches for a per-channel
activation-aware scale factor that protects the most salient channels (those with
largest activation magnitudes) from quantization error.

Both methods target 4-bit precision by default. Their 2-bit (quaternary) modes
exhibit substantially larger accuracy degradation than 4-bit, a finding consistent
across the literature. Section 5 quantifies this trade-off.

**Relevance to Q2.** GPTQ and AWQ are already noted in §D-2.4 as the canonical
examples of reconstruction quantization. Their 2-bit results set the accuracy floor
against which BQQ and similar newer methods are measured.

---

### 2.2 BQQ — Binary Quadratic Quantization (NeurIPS 2025)

BQQ (NeurIPS 2025, poster 119877) reframes 2-bit Post-Training Quantization (PTQ)
as a **binary quadratic programme** over a structured codebook. Rather than
minimising $\|W - \hat{W}\|_F^2$ with a uniform grid, BQQ decomposes $\hat{W}$ as
a product of two binary matrices $B_1, B_2 \in \{-1, +1\}^{m \times r}$ and a
learned scale $\alpha$:

$$\hat{W} \approx \alpha \cdot B_1 B_2^{\top}$$

This factorisation implicitly covers all four levels: the product $B_1 B_2^\top$
takes values in $\{-r, \ldots, -1, 0, 1, \ldots, r\}$, and the rank $r$ controls
the expressiveness of the codebook. At $r = 1$ this collapses to a rank-1
outer product; increasing $r$ trades memory for accuracy.

**Reported results.** BQQ achieves a 2.2-point improvement over previous
state-of-the-art on ImageNet for 2-bit PTQ of ResNet-class models, and shows
strong results on language tasks. It is consistently superior to GPTQ and AWQ at
the 2-bit level.

**Relevance to Q2.** BQQ's core finding — that structure in the codebook
outperforms a uniform grid under the same bit budget — resonates with Q2's
motivation: the $\mathbb{Z}_4$ alphabet is chosen not for reconstruction accuracy
but for its algebraic properties (Lee metric, complement involution, Gray map). Both
BQQ and Q2 argue that the alphabet should be chosen to match the structure of the
problem, not merely to minimise $\|W - \hat{W}\|_F^2$.

The factored-binary view also provides an arithmetic observation relevant to Q2:
the four quaternary levels can be generated by two binary decisions. Q2's Gray
encoding makes exactly this decomposition ($g = \text{sym} \oplus (\text{sym} \gg 1)$
produces two independent bits), and BQQ independently arrives at the same idea
from a reconstruction perspective.

**Key distinction.** BQQ's optimisation target is reconstruction fidelity.
Q2's is structural preservation. BQQ would not be an appropriate drop-in for the
transition-key use case: a BQQ-compressed weight matrix has no meaningful Lee
distance between its quantised entries, and the complement involution is not
preserved by the binary factorisation.

---

### 2.3 QUAD — Quantization and Parameter-Efficient Tuning for LLMs

QUAD (arxiv:2503.19353) is a PyTorch + Hugging Face Transformers framework that
combines quaternary (2-bit) weight quantization with parameter-efficient fine-tuning
(PEFT). It quantises model weights to 4 levels and adds trainable low-rank adapters
(LoRA-style) to recover accuracy lost during quantization. Key design choices:

- **Symmetric uniform codebook.** Weights are quantised to
  $\{-3\Delta, -\Delta, +\Delta, +3\Delta\}$ for learned scale $\Delta$, giving
  4 equally spaced levels around zero.
- **Joint optimisation.** QUAD trains the adapter weights while keeping the
  quantised weights frozen, using the straight-through estimator (STE) for
  gradients through the quantization step.
- **Hardware-aware packing.** Symbols are packed four per byte (2 bits per
  weight) for efficient memory layout.

**Relevance to Q2.** QUAD's symmetric 4-level codebook
$\{-3, -1, +1, +3\} \cdot \Delta$ is not the same as Q2's equiprobable threshold
$\tau^* = 0.6745 / \sqrt{n}$ scheme (§D-2.5). QUAD's levels are evenly spaced for
reconstruction; Q2's levels are equiprobable for information maximisation. For
Gaussian activations, an even spacing misallocates levels: the outer levels
$\pm 3\Delta$ will be used far less often than the inner levels $\pm\Delta$, wasting
one bit's worth of entropy. Q2's threshold maximises $I(v_i; q(v_i)) = \log_2 4 = 2$ bits
per dimension by construction.

QUAD also does not use the Lee metric or the Gray encoding. Its packed byte layout
treats the four symbols as arbitrary indices, not as elements of $\mathbb{Z}_4$.
Complement structure is absent.

**Borrowed insight.** QUAD's joint optimisation (frozen quantized weights +
trainable adapters) is relevant to the fine-tuning case of Q2-indexed models:
if a downstream task requires domain adaptation, a LoRA-style adapter over a
Q2-indexed backbone would incur only adapter parameter cost without re-running
the full quantization pipeline. This is speculative but consistent with QUAD's
findings that adapter-based recovery is efficient at 2-bit precision.

---

### 2.4 QuES — Quantized Expert Scaling

QuES (arxiv:2602.03120) targets a specific failure mode of 2-bit quantized LLMs:
degraded **arithmetic reasoning**. On tasks like GSM8K, 2-bit models (GPTQ, AWQ)
collapse to near-random performance even when general language benchmarks remain
acceptable. QuES addresses this by identifying "reasoning experts" — attention heads
and FFN channels disproportionately active during arithmetic tasks — and applying
higher precision or larger adapter capacity to those channels selectively.

This is a form of **mixed-precision quantization** targeted by an oracle derived
from task-specific activation statistics.

**Relevance to Q2.** QuES demonstrates that 2-bit precision is not uniformly harmful
across a model: some components tolerate it well, others do not. This empirical
finding independently supports Q2's §D-4.3 argument for mixed-precision quantization
guided by structural criteria. Q2 uses the Geode factorization and polytope formula
as the structural oracle; QuES uses task-specific activation statistics. The two
oracles are orthogonal but compatible.

**Borrowed insight.** QuES's finding that the failure mode of low-precision
quantization is task-specific (not uniform) suggests that Q2's transition key could
serve as a soft mixed-precision indicator: tokens whose quantization produces long
runs (low transition density, §D-3.6) are likely in low-variance, well-behaved
activation regimes, while tokens with short runs (high transition density) correspond
to higher-variance, potentially reasoning-critical activations that merit finer
quantization. This is an empirical prediction that could be tested against QuES's
identified reasoning-expert channels.

---

### 2.5 NVFP4 — 4-bit NV Float

NVFP4 is NVIDIA's 4-bit floating-point format, used in Hopper and Blackwell
architecture tensor cores. It is not a 2-bit quaternary scheme but a standard
4-bit format with one sign bit, two exponent bits, and one mantissa bit, giving
16 representable levels ($2^4$). It is included here because several papers use it
as the accuracy baseline against which 2-bit schemes are compared.

The NVFP4 vs. 2-bit comparison is consistently unfavorable for 2-bit: NVFP4
achieves near-fp16 accuracy on standard benchmarks, while 2-bit methods (including
BQQ, the current state-of-the-art) show measurable but acceptable degradation on
language tasks and more significant degradation on reasoning tasks (cf. QuES §2.4).

**Relevance to Q2.** Q2 does not compress model weights at all; it quantizes
**activations** (hidden-state vectors at inference time) into a transition key for
retrieval. The NVFP4 vs. 2-bit comparison is therefore not directly applicable. The
relevant comparison for Q2's activation quantization is retrieval quality (recall at
k, distance preservation) rather than perplexity or task accuracy.

---

## 3 Alternative Low-Precision Schemes

### 3.1 BitNet — 1.58-bit ternary architectures

BitNet b1.58 (Ma et al. 2024, arxiv:2402.17764) trains transformer models from
scratch with weights constrained to $\{-1, 0, +1\}$, achieving per-weight entropy
of $\log_2 3 \approx 1.58$ bits. This is the ternary quantization regime discussed
in §D-2.3.

Unlike post-training quantization, BitNet applies quantization during training, using
the straight-through estimator to propagate gradients through the discrete constraint.
Reported results show near-full-precision accuracy on language benchmarks at 3B and
7B parameter scales, an impressive result for ternary precision.

**Comparison with Q2 quaternary.** BitNet demonstrates that ternary precision is
achievable with minimal accuracy loss — if the model is trained with the constraint
from the start. §D-2.3 identifies the mathematical reason Q2 does not use the ternary
alphabet: $\mathbb{Z}_3$ admits no fixed-point-free involution, so the complement
structure required for the Lee metric and hairpin detection (§D-2.8, §D-3.1) is
unavailable.

BitNet's ternary weights can be trained efficiently precisely because the constraint
is baked into the forward pass. Q2 applies quaternary quantization to **activations**
at inference time, not to weights. The two problems have different constraints:
BitNet relaxes weight precision while maintaining activation precision; Q2 maintains
weight precision while compressing activation geometry for indexing.

**The BitNet activation distribution.** A notable implication: if Q2 were applied on
top of a BitNet model, the activation distribution might be non-Gaussian (because
ternary weights combined with ReLU or SiLU activations produce a distinct
distribution shape). The threshold $\tau^* = 0.6745 / \sqrt{n}$ assumes
$\mathcal{N}(0, 1/n)$ activations (§D-2.5). Empirical recalibration of $\tau^*$ for
BitNet activations would be warranted — a direction consistent with QuES's finding
that activation distributions vary significantly across model families.

---

### 3.2 Binary and ternary weight networks

Earlier work on extremely low-precision networks (BinaryConnect, XNOR-Net,
TWN/TBN) established that binary weights ($\{-1, +1\}$) are trainable with careful
initialisation and learning-rate scheduling. Quaternary weight networks (4 levels)
consistently outperform binary ones at the same memory budget because the additional
expressive power reduces the required hidden-layer width.

**Relevance to Q2.** Q2's choice of 4 levels for activation quantization is
consistent with the empirical finding that the jump from binary to quaternary
provides the largest marginal gain per additional bit. Going from 4 to 8 levels
yields diminishing returns; going from 2 to 4 levels recovers the magnitude-class
information (near/far from the threshold) that is most diagnostically valuable for
retrieval.

---

## 4 Domain-Specific Applications

### 4.1 Edge and embedded hardware

Quaternary quantization is a practical choice for deploying neural networks on
resource-constrained hardware. At 2 bits per weight, a 7B-parameter model fits in
approximately 1.75 GB — within the LPDDR budget of a mid-range smartphone. Several
papers report successful deployment of ResNet and MobileNet variants on ARM
Cortex-M microcontrollers using 2-bit quantized weights with custom SIMD packing
(4 weights per byte).

**Relevance to Q2.** Q2's thermal constraint (§D-5.3) is exactly this scenario: the
LLM is already running on the device, consuming most of the thermal budget. The Q2
transition key construction adds negligible compute on top of the already-running
LLM (one pass of L2 normalisation, thresholding, and run-reduction over the last
token's hidden state). The 2-bit packing of 32 transitions into a 64-bit integer
(§D-3.2) is a direct application of the same hardware-friendly packing used in
edge quantization literature.

**Borrowed insight.** Edge quantization implementations use compile-time-known
packing constants to enable vectorised comparisons. The Q2 threshold
$\tau^* \approx 0.6745 / \sqrt{n}$ can similarly be computed once at model-load
time and broadcast as a vector constant, allowing the quantization step to run as a
single vectorised comparison on ARM NEON or WebAssembly SIMD, with no
per-activation branch. The current WebAssembly implementation in `src/q2.wat`
already benefits from this structure.

---

### 4.2 RNNs and sentiment analysis

Early work on quaternary schemes for recurrent neural networks (pre-2020) explored
4-level weight quantization in LSTM and GRU architectures applied to sentiment
analysis benchmarks (SST-2, IMDb). These studies found that:

1. Quaternary RNN weights generalise better than binary weights on long-sequence
   tasks, because the magnitude class (near/far from threshold) carries sequence
   length information relevant to gating decisions.
2. The transition between hidden states in a quaternary-weight RNN corresponds to
   a 4-symbol trajectory in the weight-space lattice, analogous to Q2's
   transition key.

**Relevance to Q2.** The RNN finding that trajectory information is valuable at
the quaternary level independently corroborates Q2's central hypothesis: the
*sequence of quantization transitions* (the run-reduced key of §D-3.1) carries
richer structural information than any single quantized value. The RNN literature
arrived at this conclusion from a weight-quantization angle; Q2 arrives at it from
an activation-quantization angle.

---

### 4.3 CNN accelerators for industrial monitoring

A hardware accelerator study (Sensors (MDPI), 2023) implemented quaternary-weight
CNNs for real-time bearing fault diagnosis, reporting:

- **89% reduction in memory demand** relative to full-precision baseline.
- **96.37% classification accuracy** maintained, a 0.2-percentage-point drop
  from fp32.

This case is notable because bearing fault signals are periodic and low-dimensional
(vibration sensor, 1D signal), quite different from the high-dimensional activation
spaces addressed by Q2. Yet the result illustrates that quaternary quantization can
achieve near-lossless compression for structured signals.

**Relevance to Q2.** The bearing fault case is an instance where the activation
distribution is highly non-Gaussian (periodic signals produce bimodal or harmonic
distributions). The study uses a fixed symmetric codebook rather than equiprobable
thresholds. This is the exact scenario where Q2's empirical threshold calibration
(§D-2.5: reservoir sample of 1024 activations per compaction cycle) adds value over
a fixed codebook: by tracking the empirical quartiles, Q2's thresholds adapt to
non-Gaussian distributions without requiring knowledge of the distribution shape.

---

### 4.4 Quaternary neural belief propagation (BP4) for QLDPC codes

BP4 extends neural belief propagation decoders for quantum Low-Density Parity-Check
(QLDPC) codes from binary to quaternary alphabets. QLDPC error correction requires
passing messages over GF(4) (the field with 4 elements), which has a different
algebraic structure from $\mathbb{Z}_4$ (the ring used in Q2). Specifically, GF(4)
uses XOR-based multiplication; $\mathbb{Z}_4$ uses modular arithmetic. The message
passing operates on syndrome vectors in GF(4), not on geometry-preserving lattice
codes.

**Relevance to Q2.** BP4 and Q2 both use 4-symbol alphabets but are built on
different algebraic structures. BP4 requires GF(4) for its syndrome arithmetic;
Q2 requires $\mathbb{Z}_4$ for the Lee metric and complement involution. These
are not interchangeable: the Hamming, Lee, and GF(4) metrics have different distance
geometries, and the error-correction properties of Kerdock and Preparata codes
(§P-9) do not apply to GF(4)-based QLDPC codes.

The BP4 work is included here for completeness and to confirm that the quaternary
alphabet finds natural expression in multiple algebraic settings, each tailored to
its geometric context.

---

### 4.5 Image steganography (OPMS-QQGE)

OPMS-QQGE (Optimal Phase-based Multi-Scale Quaternary Quantized Gaussian Embedding)
applies quaternary quantization to the frequency domain of images for steganographic
embedding. The quantization step maps DCT coefficients to 4 levels, and the
embedding modulates inter-coefficient phase relationships at each level. This
achieves high payload capacity and resistance to CNN-based steganalyzers.

**Relevance to Q2.** The steganographic use case is superficially different but
shares a deep structural property: OPMS-QQGE exploits the *relational geometry*
of the quantized space (inter-coefficient phase differences) rather than the
absolute values of individual coefficients. This is precisely the distinction Q2
makes in §D-2.4: structural quantization preserves relations, not values.

OPMS-QQGE's finding that a 4-level representation provides sufficient degrees of
freedom for robust phase-relationship encoding parallels Q2's finding that 4 levels
provide the minimum alphabet for the complement involution. Both arrive at 4 from a
relational-geometry requirement rather than a reconstruction-accuracy requirement.

---

## 5 Accuracy vs. Efficiency Trade-offs Across the Literature

The literature presents a consistent picture of the accuracy-efficiency frontier for
2-bit quantization:

| Method | Bits | Setting | Key result |
|--------|------|---------|------------|
| GPTQ (2-bit) | 2 | PTQ, LLM weights | Substantial perplexity increase; ~70% of 4-bit accuracy |
| AWQ (2-bit) | 2 | PTQ, LLM weights | Similar to GPTQ; activation-aware scaling helps ~1-2 pp |
| BQQ | 2 | PTQ, vision + language | +2.2 pp over GPTQ on ImageNet; best published PTQ at 2-bit |
| QUAD | 2 | QAT + PEFT, LLM weights | Recovery via adapters; near 4-bit accuracy on GLUE |
| QuES | 2 | Fine-tuning, arithmetic | Targeted recovery on GSM8K; general tasks unaffected |
| BitNet b1.58 | 1.58 | QAT from scratch, LLM weights | Near-fp16 on language benchmarks at 3B-7B params |
| NVFP4 | 4 | Hardware format, LLM weights | Near-fp16 reference; 2-bit methods compared against this |
| Edge CNN (bearing) | 2 | QAT, small CNN | 96.4% accuracy, 89% memory reduction |

**Key findings from the trade-off landscape:**

1. **2-bit PTQ is feasible but imperfect.** No method recovers full fp32 accuracy
   without either training (QAT) or fine-tuning (adapters). BQQ is the best
   published PTQ result as of 2025.

2. **Reasoning degrades disproportionately.** Language fluency is more robust to
   2-bit precision than structured reasoning. QuES directly targets this gap.

3. **QAT from scratch is qualitatively different.** BitNet demonstrates that
   training with the quantization constraint from the start achieves a different
   (better) accuracy-efficiency trade-off than post-training compression.

4. **Task and domain matter.** Industrial classification (bearing fault) shows
   near-lossless 2-bit compression; open-ended language generation shows
   measurable degradation. The distribution shape and task difficulty interact.

**Where Q2 sits.** Q2 does not compress model weights; it quantizes activations
for indexing. The accuracy metric is retrieval quality, not perplexity. Q2 makes
no accuracy-efficiency trade-off on the model's generative performance — the LLM
runs at full precision. The trade-off it makes is between index compactness (64-bit
key vs. full float32 embedding) and retrieval fidelity (transition-key recall vs.
cosine-similarity recall). These are distinct dimensions from the weight-quantization
trade-offs surveyed above.

---

## 6 Key Distinctions: Q2 vs. the Field

The following table summarises how Q2 differs from the main classes of related work
along the axes that matter most:

| Dimension | Reconstruction methods (BQQ/GPTQ/QUAD) | Q2 structural quantization |
|-----------|----------------------------------------|-----------------------------|
| **What is quantized** | Model weights | Inference-time activations |
| **Objective** | Minimize reconstruction error | Preserve relational geometry |
| **Metric** | Frobenius norm | Lee distance on $\mathbb{Z}_4$ |
| **Alphabet design** | Minimize $\|W - \hat{W}\|$ | Equiprobable, complement-closed |
| **Output** | Quantized weight matrix | 64-bit transition key |
| **Use case** | Memory-efficient inference | Compact retrieval index |
| **Evaluation** | Perplexity, task accuracy | Recall, distance preservation |
| **Algebraic structure** | Varies (grid, factored-binary, etc.) | $\mathbb{Z}_4$, Lee metric, Gray map |
| **Complement involution** | Not required | Required (§D-2.8) |
| **Cross-model invariance** | Not targeted | Targeted (§D-5.4) |

The single most important distinction is the **target of quantization**: the
reconstruction methods quantize weights to save memory at inference time; Q2
quantizes activations to produce a compact retrieval index. They solve different
problems with the same alphabet.

---

## 7 Borrowed Insights

The following findings from the literature have direct actionable implications for Q2:

### 7.1 Factored-binary codebook (from BQQ)

BQQ's result that two binary decisions generate the four quaternary levels more
efficiently than a uniform grid confirms Q2's Gray encoding choice ($g = \text{sym}
\oplus (\text{sym} \gg 1)$). The two bits of the Gray code are algebraically
independent, which is exactly what BQQ's binary factorisation achieves. This
provides an independent theoretical justification for Q2's Gray map from a
reconstruction-error perspective.

### 7.2 Equiprobable thresholds outperform equal-spacing (from QUAD contrast)

QUAD uses equal-spacing for its 4-level codebook. Q2's equiprobable threshold
$\tau^*$ maximises entropy per dimension ($I = 2$ bits). The literature on
information-theoretic quantization (Max-Lloyd algorithm) confirms that for Gaussian
sources, equiprobable thresholds minimise entropy-normalised distortion. Q2's design
is optimal by this criterion. QUAD's equal-spacing design is suboptimal for Gaussian
activations — an insight that supports Q2's threshold design rather than suggesting
a change.

### 7.3 Mixed-precision oracle from activation statistics (from QuES)

QuES uses task-specific activation statistics to identify high-importance channels
that need higher precision. Q2's transition density (§D-3.6) is an activation
statistic: low-density windows correspond to low-variance, "settled" activations;
high-density windows correspond to high-variance, structurally active regions. This
suggests using transition density as a lightweight proxy for "quantization
sensitivity" — high-density tokens are candidates for finer quantization or
auxiliary full-precision embedding.

This is speculative but empirically testable. It extends P17 (§D-4.3) with a
concrete mechanism borrowed from QuES's methodology.

### 7.4 Per-channel scale factors (from AWQ and GPTQ)

AWQ's activation-aware scale factor protects high-activation channels from
quantization error by rescaling before quantization and inverse-rescaling after.
Q2's L2 normalisation step (§D-README) achieves a similar effect at the vector
level: by normalising to unit length before thresholding, Q2 removes the global
scale, ensuring that the threshold $\tau^*$ is applied to a distribution with unit
variance rather than an uncalibrated raw activation. Per-channel scale factors
(AWQ-style) within the normalised vector are not currently applied; this could be
considered as an extension if per-channel variance is found to be non-uniform after
normalisation.

### 7.5 Trajectory information in RNNs (from RNN quaternary literature)

The RNN finding that sequence-of-transitions carries richer information than
individual quantized values (§4.2 above) independently confirms Q2's run-reduction
hypothesis. Both approaches observe that the temporal or sequential pattern of
transitions through the quantization grid — not the individual cell assignments —
is the primary carrier of structural information.

### 7.6 Adapter-based recovery for downstream tasks (from QUAD)

QUAD demonstrates that LoRA-style adapters efficiently recover task-specific
accuracy on top of frozen 2-bit quantized weights. For the Q2 use case, an analogous
pattern exists: the transition key captures the base-model's activation geometry;
domain adaptation for a downstream retrieval task could be achieved by training a
small adapter that modifies the last-hidden-state before Q2 quantization, rather
than retraining or recalibrating the full Q2 index. This is consistent with QUAD's
finding that adapters are highly parameter-efficient at 2-bit precision.

---

## References

- BQQ: Binary Quadratic Quantization. NeurIPS 2025, poster 119877.
  <https://neurips.cc/virtual/2025/poster/119877>
- QUAD: Quantization and Parameter-Efficient Tuning for LLMs. arxiv:2503.19353.
  <https://arxiv.org/html/2503.19353v1>
- QuES: Quantized Expert Scaling. arxiv:2602.03120.
  <https://arxiv.org/html/2602.03120v1>
- BitNet b1.58: arxiv:2402.12263.
  <https://arxiv.org/html/2402.12263v2>
- OPMS-QQGE steganography survey: arxiv:2509.13514.
  <https://arxiv.org/html/2509.13514v1>
- IEEE Sensors J. CNN accelerator for bearing fault diagnosis. Vol. 23, no. 13,
  2023. <https://www.mdpi.com/1424-8220/23/13/5897>
- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate
  Post-Training Quantization for Generative Pre-Trained Transformers. arxiv:2210.17323.
- Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). AWQ:
  Activation-Aware Weight Quantization for LLM Compression and Acceleration.
  arxiv:2306.00978.
- Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R.,
  Wei, F., & Wei, F. (2024). The Era of 1-bit LLMs: All Large Language Models are
  in 1.58 Bits. arxiv:2402.17764.
- Hammons, A. R., Kumar, P. V., Calderbank, A. R., Sloane, N. J. A., & Solé, P.
  (1994). The $\mathbb{Z}_4$-linearity of Kerdock, Preparata, Goethals, and related
  codes. *IEEE Trans. Inform. Theory* 40:2, 301--319.
- Wildberger, N. J. & Rubine, D. (2025). A Hyper-Catalan Series Solution to
  Polynomial Equations, and the Geode. *Amer. Math. Monthly* 132:5, 383--402.
  DOI: 10.1080/00029890.2025.2460966
