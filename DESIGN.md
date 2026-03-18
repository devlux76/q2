# Quaternary Semantic Quantization: Design

> **Related documents:** [PREDICTIONS.md](PREDICTIONS.md) · [TESTING.md](TESTING.md)

---

## Contents

1. [The Embedding Problem](#1-the-embedding-problem)
   - 1.1 [Embeddings and the unit hypersphere](#11-embeddings-and-the-unit-hypersphere)
   - 1.2 [Incommensurability](#12-incommensurability)
   - 1.3 [Concentration of measure](#13-concentration-of-measure)
   - 1.4 [The surface-area information bound](#14-the-surface-area-information-bound)
   - 1.5 [The L1 metric and the cross-polytope](#15-the-l1-metric-and-the-cross-polytope)
   - 1.6 [The quaternary coordinate](#16-the-quaternary-coordinate)
   - 1.7 [Index design consequences](#17-index-design-consequences)
2. [Quantization](#2-quantization)
   - 2.1 [The thermal constraint](#21-the-thermal-constraint)
   - 2.2 [Binary quantization](#22-binary-quantization)
   - 2.3 [Ternary quantization](#23-ternary-quantization)
   - 2.4 [Standard Q4 (different problem)](#24-standard-q4-different-problem)
   - 2.5 [Quaternary semantic quantization](#25-quaternary-semantic-quantization)
   - 2.6 [The Lee metric](#26-the-lee-metric)
   - 2.7 [The Gray map](#27-the-gray-map)
   - 2.8 [The complement involution](#28-the-complement-involution)
3. [The Transition Key](#3-the-transition-key)
   - 3.1 [Run-reduction](#31-run-reduction)
   - 3.2 [The 64-bit integer key](#32-the-64-bit-integer-key)
   - 3.3 [MSB alignment and prefix semantics](#33-msb-alignment-and-prefix-semantics)
   - 3.4 [Block file organisation](#34-block-file-organisation)
   - 3.5 [Window queries](#35-window-queries)
   - 3.6 [Bucket density](#36-bucket-density)

---

## 1 The Embedding Problem

### 1.1 Embeddings and the unit hypersphere

A transformer model maps a document to a vector in $\mathbb{R}^n$ by passing its token
sequence through the network, mean-pooling activations over token positions, and
L2-normalising the result. The output is a point on the unit hypersphere:

$$e \in S^{n-1} = \left\{ x \in \mathbb{R}^n : \|x\| = 1 \right\}$$

Each coordinate satisfies $\sum_i e_i^2 = 1$. Semantic similarity between documents is
measured by cosine similarity, which equals the dot product between normalised vectors:

$$\text{sim}(u, v) = u \cdot v = \cos\theta_{uv}$$

where $\theta_{uv}$ is the angle between $u$ and $v$.

---

### 1.2 Incommensurability

$S^{n-1}$ has no preferred orientation. Every rotation $Q \in O(n)$ is an isometry:

$$u \cdot v = (Qu) \cdot (Qv)$$

A model trained on a corpus places semantically similar documents near each other on
the sphere, but the absolute coordinate frame is determined by random initialisation
and training order. A second model trained on the same corpus produces an embedding
space related to the first by some unknown $Q \in O(n)$:

$$e_B(x) \approx Q \cdot e_A(x)$$

$Q$ is not available from the model weights and cannot be recovered without a large set
of paired examples and a Procrustes alignment solve.

For any two documents $x, y$ embedded by different models:

$$e_A(x) \cdot e_B(y)$$

depends entirely on $Q$. The dot product measures alignment between two arbitrary
coordinate frames, not semantic similarity between documents.

An untrained transformer also produces vectors on $S^{n-1}$, but with no training
signal to cluster similar documents, the geometry carries no semantic content. An
embedding is meaningful as a retrieval key only after training.

---

### 1.3 Concentration of measure

The volume of the $n$-dimensional unit ball:

$$V_n(1) = \frac{\pi^{n/2}}{\Gamma\!\left(\tfrac{n}{2}+1\right)}$$

satisfies $V_n(1) \to 0$ as $n \to \infty$ (Stirling's approximation gives
$V_n(1) \sim \frac{1}{\sqrt{\pi n}}\left(\frac{2\pi e}{n}\right)^{n/2}$).

The fraction of the unit ball's volume contained in the outer shell of thickness
$\varepsilon$ is:

$$f_{\text{shell}}(n, \varepsilon) = 1 - (1-\varepsilon)^n$$

For any fixed $\varepsilon > 0$, $f_{\text{shell}} \to 1$ as $n \to \infty$. The shell
thickness required to capture fraction $f$ is:

$$\varepsilon^*(f, n) = 1 - (1-f)^{1/n} \approx \frac{-\ln(1-f)}{n}$$

| Fraction captured | Shell thickness |
|:-----------------:|:---------------:|
| 63.2% | $1/n$ |
| 86.5% | $2/n$ |
| 95.0% | $3/n$ |
| 99.0% | $4.61/n$ |

In 256 dimensions, 99% of the ball's volume lies within a shell of radial thickness
$4.61/256 \approx 0.018$.

For $u, v$ drawn uniformly from $S^{n-1}$:

$$\mathbb{E}[u \cdot v] = 0 \qquad \operatorname{Var}(u \cdot v) = \frac{1}{n}$$

The standard deviation of cosine similarity between two random unit vectors is $1/\sqrt{n}$.
At $n = 256$ this is $\approx 0.063$: the full range $[-1, +1]$ is compressed into
noise of order $\pm 0.06$. A trained model must push similar documents closer than
this noise floor.

Each coordinate of a random unit vector satisfies:

$$\mathbb{E}[e_i] = 0 \qquad \mathbb{E}[e_i^2] = \frac{1}{n}$$

In high dimensions, the coordinates of a uniform random unit vector are approximately
i.i.d. $\mathcal{N}(0, 1/n)$.

---

### 1.4 The surface-area information bound

The shell thickness $\varepsilon^*(n) \sim c/n$ shrinks as $n$ grows. If the semantic
space has a characteristic scale $L$, the absolute shell thickness is:

$$\delta(n) = \frac{c \cdot L}{n}$$

Setting $\delta(n_{\max}) = \ell_P$ (the Planck length,
$\approx 1.616 \times 10^{-35}$ m) gives:

$$n_{\max} = \frac{c \cdot L}{\ell_P}$$

Above $n_{\max}$, additional dimensions subdivide already-indistinguishable states.

The Bekenstein–Hawking bound states that the maximum entropy of a system enclosed by
area $A$ is $S_{\max} = A / (4\ell_P^2)$ — information scales with surface area, not
volume. The same structure follows from the geometry of high-dimensional spheres:

1. All volume concentrates at the surface as $n \to \infty$.
2. All distinguishable states live on $S^{n-1}$, not in the ball.
3. The capacity of the space is determined by how finely $S^{n-1}$ can be
   partitioned — a surface quantity.

The surface area of $S^{n-1}$ is $\mathcal{A}_{n-1}(1) = 2\pi^{n/2}/\Gamma(n/2)$.
The minimum distinguishable angular separation grows as $\sim 1/\sqrt{n}$, so the
marginal information per additional dimension decreases as the shell compresses.

The semantic content of a document is encoded in which region of $S^{n-1}$ its
embedding occupies. The quantization problem is the problem of discretising that
surface efficiently.

---

### 1.5 The L1 metric and the cross-polytope

The L2 distance couples all $n$ coordinates under a square root:

$$d_2(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$

The L1 metric decouples them:

$$d_1(u, v) = \sum_{i=1}^n |u_i - v_i|$$

The L1 unit ball is the cross-polytope:

$$B_1^n = \left\{ x \in \mathbb{R}^n : \sum_{i=1}^n |x_i| \leq 1 \right\}$$

In two dimensions this is the axis-aligned diamond with vertices at $(\pm 1, 0)$ and
$(0, \pm 1)$. Its boundary satisfies $|x_1| + |x_2| = 1$. The L1 "circle" has
$\pi_1 = 4$, which is exact and rational.

The $n$-dimensional cross-polytope has $2n$ vertices at $\pm e_i$ for each standard
basis vector, and $2^n$ facets. Each coordinate contributes an independent 1D
displacement; the total L1 distance is their sum. There is no coupling between
dimensions and no transcendental constant.

The axis-aligned Cartesian grid is the natural grid for the L1 ball: the diamond's
faces lie on the coordinate halfspaces, so the grid and the ball are aligned.

---

### 1.6 The quaternary coordinate

Under the L1 metric, $d_1(u, v)$ decomposes into $n$ independent scalar problems:
for each dimension $i$, how far apart are $u_i$ and $v_i$? The full distance is their
sum.

A single coordinate $x_i \in \mathbb{R}$ has two structural features relevant to
position on the L1 unit ball: its **sign** (which side of the origin) and its
**magnitude class** (near the origin or far from it). These two binary features yield
four cells. Fewer cells collapse one feature; more cells subdivide within a cell,
encoding intra-cell position that does not contribute to the L1 distance between
cells. Four is the minimum that preserves both features.

The four cells map to $\{A, B, C, D\} \leftrightarrow \{0, 1, 2, 3\}$ in
$\mathbb{Z}_4$. The Lee metric on $\mathbb{Z}_4$:

$$d_L(u_i, v_i) = \min(|u_i - v_i|,\ 4 - |u_i - v_i|)$$

is the L1 distance on the 4-point cycle. The cyclic wrap $d_L(D, A) = 1$ reflects
that strong-negative and strong-positive are both extreme vertices of the same axis,
adjacent in the L1 cross-polytope.

Extended to vectors:

$$d_L(u, v) = \sum_{i=1}^{n} \min(|u_i - v_i|,\ 4 - |u_i - v_i|)$$

The Gray map (§2.7) makes this computable by `popcnt(XOR)` on 64-byte vectors
without symbol decoding.

**The float32 activation** is not the ground truth that the quaternary symbol
approximates. It is an overcomplete representation of an ordinal position. The bits
encoding intra-cell displacement contribute to the L2 norm and to shell concentration
(§1.3); they are not recoverable signal for L1 retrieval. The quantization discards
them.

---

### 1.7 Index design consequences

Three consequences follow from §1.1–1.6.

**Incommensurability is absolute.** The quantization thresholds are calibrated from
one model's activation distribution. A different model produces different thresholds.
Quaternary codes are not comparable across models. An index must be rebuilt when the
model changes.

**The thermal constraint and the geometric constraint coincide.** Cross-model
alignment requires a Procrustes solve over paired examples — geometrically necessary
but thermally impossible on constrained hardware. Using the LLM's own activations
satisfies both constraints simultaneously.

**The quantization problem restated.** The question is not: how many bits per
dimension approximate angular distance on $S^{n-1}$? The question is: what is the
natural discrete coordinate system of the L1 unit ball? The answer — four cells per
dimension — is derived in §2.5.

---

## 2 Quantization

### 2.1 The thermal constraint

Running a large language model on a consumer device operates near the thermal ceiling
of that hardware. A second dedicated embedding model, run in parallel or in sequence,
would exceed the thermal budget, cause throttling, and drain the battery.

The constraint is therefore: given that an LLM is already running and already producing
activations, what is the highest-quality semantic index constructible from those
activations alone, at a fixed and predictable compute cost?

---

### 2.2 Binary quantization

The simplest compression of a float32 activation $v_i$ is the sign bit:

$$q_{\text{bin}}(v_i) = \begin{cases} 0 & v_i \leq 0 \\ 1 & v_i > 0 \end{cases}$$

This produces one bit per dimension. The natural distance on the result is Hamming
distance.

**What is lost.** Float32 encodes both direction (sign) and magnitude. Binary
quantization discards magnitude entirely: $v_i = 0.01$ and $v_i = 4.7$ are
indistinguishable. For activations drawn from $\mathcal{N}(0, \sigma^2)$:

$$I(v_i;\ q_{\text{bin}}(v_i)) = H(q_{\text{bin}}(v_i)) = 1 \text{ bit}$$

The magnitude information — which for mean-pooled transformer activations is
substantial — is absent from the index.

---

### 2.3 Ternary quantization

Ternary quantization adds a central state:

$$q_{\text{tern}}(v_i) = \begin{cases} - & v_i \leq -\tau \\ 0 & -\tau < v_i \leq \tau \\ + & v_i > \tau \end{cases}$$

This recovers one additional bit: whether the activation is near the boundary or
committed. For $\tau$ chosen to make the three states equiprobable:

$$I(v_i;\ q_{\text{tern}}(v_i)) = \log_2 3 \approx 1.585 \text{ bits}$$

**What remains lost.** The ternary alphabet $\{-, 0, +\}$ has group structure
$\mathbb{Z}_3$, which admits no fixed-point-free involution — there is no map
$\theta: \mathbb{Z}_3 \to \mathbb{Z}_3$ satisfying $\theta^2 = \text{id}$ and
$\theta(x) \neq x$ for all $x$. The complement relationship between a concept and its
semantic opposite — present in the activation space as the relationship between
strong-positive and strong-negative directions — is not representable.

---

### 2.4 Standard Q4 (different problem)

Standard 4-bit quantization for LLM weight compression (GPTQ, AWQ, and related
methods) assigns 4 bits per parameter using a learned or analytical codebook that
minimises reconstruction error:

$$\min_{\hat{W}} \| W - \hat{W} \|_F^2$$

subject to $\hat{W}$ having 4-bit entries ($2^4 = 16$ levels per dimension).

This is weight quantization, not activation quantization for retrieval. The objective,
metric, and distribution are all different. The two methods are not in the same design
space.

---

### 2.5 Quaternary semantic quantization

The preceding analysis identifies what a correct scheme requires:

1. **Sign** — which side of the semantic hyperplane.
2. **Magnitude class** — near the boundary or strongly committed.
3. **Complement structure** — a fixed-point-free involution $\theta$ satisfying
   $\theta^2 = \text{id}$ and $\theta(x) \neq x$ for all $x$.
4. **Minimum alphabet** — the fewest symbols satisfying all three requirements.

Requirements 1 and 2 together demand at least four states. Requirement 3 demands a
complement involution. Requirement 4 asks whether three states suffice: with
$\{-, 0, +\}$, the only candidate involution swaps $-$ and $+$, leaving $0$ with no
valid image. Four is the minimum.

**The four states** are labelled by signed magnitude class:

$$\{A,\ B,\ C,\ D\} \;\longleftrightarrow\; \{\text{strong−},\ \text{weak−},\ \text{weak+},\ \text{strong+}\}$$

**The quantization threshold.** For activations drawn from $\mathcal{N}(0, 1/n_s)$
where $n_s$ is the embedding dimension[^1], the maximum-entropy condition requires
equiprobable states:

$$P(v_i \leq -\tau^*) = P(-\tau^* < v_i \leq 0) = P(0 < v_i \leq \tau^*) = P(v_i > \tau^*) = \tfrac{1}{4}$$

The threshold is:

$$\tau^* = \frac{\Phi^{-1}(3/4)}{\sqrt{n_s}} \approx \frac{0.6745}{\sqrt{n_s}}$$

**The quantization function:**

$$q(v_i) = \begin{cases} A & v_i \leq -\tau^* \\ B & -\tau^* < v_i \leq 0 \\ C & 0 < v_i \leq \tau^* \\ D & v_i > \tau^* \end{cases}$$

**Empirical calibration.** In practice $\tau^*$ is estimated from a reservoir sample
of 1 024 document activations per compaction cycle, using the empirical 25th and 75th
percentiles of $v_i$ to keep the symbol distribution close to equiprobable without
assuming a specific activation shape.

Under the equiprobable target, each dimension carries:

$$I(v_i;\ q(v_i)) = \log_2 4 = 2 \text{ bits}$$

**Lee distances between the four states:**

| Pair | Lee distance |
|:----:|:------------:|
| $A$–$A$, $B$–$B$, $C$–$C$, $D$–$D$ | 0 |
| $A$–$B$, $B$–$C$, $C$–$D$ | 1 |
| $D$–$A$ (cyclic wrap) | 1 |
| $A$–$C$, $B$–$D$ (complements) | 2 |

---

### 2.6 The Lee metric

Given four ordered states $\{A=0,\ B=1,\ C=2,\ D=3\}$ arranged cyclically in
$\mathbb{Z}_4$, the Lee distance is:

$$d_L(u, v) = \min(|u - v|,\ 4 - |u - v|)$$

**Why this metric matches semantic distance.** A weak-negative and a weak-positive
activation ($B$–$C$) are adjacent states: the concept was weakly activated in both
documents, with opposite sign. Lee distance 1 reflects this. The complement pairs
$A$–$C$ and $B$–$D$ represent semantic opposition: one document activates a dimension
strongly in one direction, the other in the complementary direction. Lee distance 2
reflects this. Strong-negative and strong-positive ($A$–$D$) share strong commitment
to their respective directions; the cyclic metric assigns them distance 1, not 2.

Hamming distance on raw 2-bit strings does not capture the cyclic structure; it
assigns $A$–$D$ (encodings `00` and `10`) distance 1 correctly, but $A$–$C$
(encodings `00` and `11`) distance 2 for the wrong reason (both bits flip, not
because of complement structure).

**Example.** For two 4-dimensional vectors:

$$u = [A, B, C, D] = [0, 1, 2, 3]$$
$$v = [A, C, C, B] = [0, 2, 2, 1]$$

Position-wise: $d_L(0,0)=0$, $d_L(1,2)=1$, $d_L(2,2)=0$, $d_L(3,1)=\min(2,2)=2$.
Total: $d_L(u,v) = 3$.

For $n = 256$ dimensions the maximum Lee distance is $256 \times 2 = 512$.

---

### 2.7 The Gray map

The Gray map $\phi: \mathbb{Z}_4 \to \{0,1\}^2$ is the unique binary encoding that
makes Hamming distance on the encoded vectors equal to Lee distance on the originals:

$$\phi(0) = \texttt{00} \qquad \phi(1) = \texttt{01} \qquad \phi(2) = \texttt{11} \qquad \phi(3) = \texttt{10}$$

**Theorem 2.1** *(Hammons, Kumar, Calderbank, Sloane, Solé, 1994).* $\phi$ is an
isometry from $(\mathbb{Z}_4^n, d_L)$ to $(\{0,1\}^{2n}, d_H)$:

$$d_H(\phi(u), \phi(v)) = d_L(u, v) \quad \text{for all } u, v \in \mathbb{Z}_4^n$$

**Proof of the single-symbol case.**

| Pair | Encodings | XOR | Hamming | Lee |
|:----:|:---------:|:---:|:-------:|:---:|
| $(0,0),(1,1),(2,2),(3,3)$ | equal | `00` | 0 | 0 |
| $(0,1),(1,2),(2,3)$ | adjacent | `01` or `10` | 1 | 1 |
| $(0,3)$ | cyclic wrap | `10` | 1 | $\min(3,1)=1$ |
| $(0,2),(1,3)$ | complement | `11` | 2 | $\min(2,2)=2$ |

Hamming equals Lee in every case. The extension to vectors follows by summing over
positions. $\square$

**Consequence.** `popcnt(XOR)` on Gray-encoded vectors computes the exact cyclic Lee
distance without any symbol-level decoding. This is the fastest hardware-accelerated
distance primitive available.

**Closed form.** For $n \in \{0,1,2,3\}$:

$$\phi(n) = n \oplus (n \gg 1)$$

This allows computing Gray codes for four symbols simultaneously in one SIMD XOR
instruction.

**The encoding table:**

| Symbol | Meaning | $\mathbb{Z}_4$ | $\phi$ |
|:------:|:-------:|:--------------:|:------:|
| $A$ | Strong negative | 0 | `00` |
| $B$ | Weak negative | 1 | `01` |
| $C$ | Weak positive | 2 | `11` |
| $D$ | Strong positive | 3 | `10` |

A 256-dimensional embedding encodes to $256 \times 2 = 512$ bits = 64 bytes.

---

### 2.8 The complement involution

Define $\theta: \mathbb{Z}_4 \to \mathbb{Z}_4$ by:

$$\theta(x) = x + 2 \pmod{4}$$

This gives $\theta(A)=C$, $\theta(B)=D$, $\theta(C)=A$, $\theta(D)=B$.

In the $\phi$-encoding, $\theta$ is bitwise NOT: $\phi(\theta(x)) = \overline{\phi(x)}$.

**Proposition 2.2** *(Universal complement identity).* For any vector
$v \in \mathbb{Z}_4^{256}$ and its complement $\bar{v}$ defined componentwise by $\theta$:

$$\phi(v) \oplus \phi(\bar{v}) = \underbrace{\texttt{FF}\ldots\texttt{FF}}_{64 \text{ bytes}}$$

*Proof.* For any 2-bit Gray encoding $e$, $e \oplus \bar{e} = \texttt{11}$ by
bitwise NOT. All 256 positions contribute `11`; concatenated over 512 bits the result
is all-ones. $\square$

**Corollary 2.3** *(Self-complement exclusion).* No vector $v$ satisfies $v = \bar{v}$.

*Proof.* If $v = \bar{v}$ then $\phi(v) \oplus \phi(\bar{v}) = 0 \neq \texttt{FF}\ldots\texttt{FF}$. $\square$

---

## 3 The Transition Key

### 3.1 Run-reduction

Given a quantized vector $V = (v_0, v_1, \ldots, v_{n-1}) \in \{0,1,2,3\}^n$,
**run-reduction** produces a transition sequence by a single left-to-right pass:

```
R ← (v₀)
for i in 1..n-1:
    if vᵢ ≠ vᵢ₋₁: append vᵢ to R
```

The result $R = (r_0, r_1, \ldots, r_{k-1})$ is the sequence of distinct consecutive
values of $V$: every run of identical adjacent symbols is collapsed to its first
element.

**Lemma 3.1** *(Idempotence).* $\text{reduce}(\text{reduce}(V)) = \text{reduce}(V)$.

*Proof.* In $R = \text{reduce}(V)$, adjacent elements are always distinct by
construction. The second pass therefore appends every element of $R$, returning $R$
unchanged. $\square$

**Lemma 3.2** *(Length bound).* $|R| \leq |V|$, with equality iff $V$ is already a
transition sequence.

*Proof.* Each element of $V$ contributes at most one element to $R$. $\square$

**Lemma 3.3** *(Run-length invariance).* Two vectors $V$ and $V'$ with the same
transition sequence but different run lengths map to the same $R$.

*Proof.* Run-reduction discards run-length information. $\square$

Run-length invariance means a document that visits a semantic state once and a
document that dwells in it for many consecutive dimensions share the same key. The key
records which states were visited, not how long each visit lasted.

---

### 3.2 The 64-bit integer key

Given $R = (r_0, r_1, \ldots, r_{k-1})$ with $r_i \in \{0,1,2,3\}$, the integer key
$K$ is the base-4 number with $R$ as its digits, most significant first:

$$K(R) = \sum_{i=0}^{\min(k,32)-1} r_i \cdot 4^{31-i}$$

**Theorem 3.4** *(Exact 64-bit fit).* The range of $K$ is $[0,\ 2^{64}-1]$.

*Proof.* The maximum 32-digit base-4 number is:

$$K_{\max} = \sum_{i=0}^{31} 3 \cdot 4^i = 3 \cdot \frac{4^{32}-1}{3} = 4^{32} - 1$$

Since $4 = 2^2$, we have $4^{32} = 2^{64}$, so $K_{\max} = 2^{64} - 1$. The minimum
is 0. No overflow, no wasted bit. $\square$

**Corollary 3.5** *(Bit layout).* Bits 63–62 encode $r_0$; bits 61–60 encode $r_1$;
bits 1–0 encode $r_{31}$.

**Sequences longer than 32.** If $|R| > 32$, only the first 32 symbols enter the key.
Documents whose transition sequences agree in the first 32 steps and diverge only
afterward share a key and are retrieved as a group by any window query. Intra-bucket
distinction is resolved by the Lee-distance re-ranking step.

---

### 3.3 MSB alignment and prefix semantics

The key is left-aligned: $r_0$ occupies the most significant two bits.

**Definition.** The *semantic depth* of a transition at position $i$ in $R$ is $i$.
Depth 0 is the first transition; depth 31 is the finest resolvable discrimination.

**Proposition 3.6** *(Prefix clustering).* Two keys $K_1$ and $K_2$ share a common
prefix of length $j$ if and only if their transition sequences agree in positions
$0$ through $j-1$.

*Proof.* The $j$ most significant base-4 digits of $K$ are exactly $r_0, \ldots,
r_{j-1}$. $\square$

**Corollary 3.7** *(Key distance bounds).* If $R_1$ and $R_2$ first diverge at
position $j$:

$$|K_1 - K_2| \leq 4^{32-j} - 1$$

A window of half-width $\delta = 4^{32-j}$ recovers all sequences that agree with the
query in the first $j$ transitions.

---

### 3.4 Block file organisation

The 64-bit key space $[0, 2^{64})$ is partitioned into 8 block files, each covering
$2^{61}$ consecutive keys.

**Partition.** Block $b \in \{0, \ldots, 7\}$ covers:

$$\mathcal{B}_b = \left[ b \cdot 2^{61},\ (b+1) \cdot 2^{61} \right)$$

The block index for a key $K$ is the top 3 bits: $b(K) = K \gg 61$.

| Block | Hex range |
|:-----:|:----------|
| 0 | `0x0000000000000000` – `0x1FFFFFFFFFFFFFFF` |
| 1 | `0x2000000000000000` – `0x3FFFFFFFFFFFFFFF` |
| 2 | `0x4000000000000000` – `0x5FFFFFFFFFFFFFFF` |
| 3 | `0x6000000000000000` – `0x7FFFFFFFFFFFFFFF` |
| 4 | `0x8000000000000000` – `0x9FFFFFFFFFFFFFFF` |
| 5 | `0xA000000000000000` – `0xBFFFFFFFFFFFFFFF` |
| 6 | `0xC000000000000000` – `0xDFFFFFFFFFFFFFFF` |
| 7 | `0xE000000000000000` – `0xFFFFFFFFFFFFFFFF` |

Within each block file, keys are stored sorted. A block file is a sorted sequence of
$(K, \textit{doc\_id})$ pairs. Range queries reduce to a binary-search lower bound
followed by a sequential scan to the upper bound.

**Block count.** $N_b = 8$ balances two constraints:

- *Density:* each file must contain enough entries ($C/N_b$ on average) for binary
  search to be worthwhile.
- *Sparsity:* a window $[K-\delta, K+\delta]$ must select a small result set.
  Block size $2^{61}$ with $C \leq 10^{12}$ documents gives occupancy
  $\leq 10^{12}/2^{61} \approx 4 \times 10^{-7}$.

$N_b = 8 = 2^3$ keeps block routing to a 3-bit shift. Fewer blocks merge too many
concepts into one file; more blocks fragment the corpus into mostly-empty files.

---

### 3.5 Window queries

**Definition.** A *window query* with centre $K$ and half-width $\delta$ retrieves all
indexed entries with key in $[K-\delta,\ K+\delta]$:

$$W(K, \delta) = \{ (K',\ \textit{doc\_id}) : |K' - K| \leq \delta \}$$

**Theorem 3.8** *(Two-file bound).* For any $K$ and $\delta < 2^{61}$, $W(K, \delta)$
intersects at most 2 block files.

*Proof.* The interval $[K-\delta, K+\delta]$ has width $2\delta+1 \leq 2^{61}$. A
contiguous interval of that width straddles at most one block boundary. $\square$

Any practically chosen window satisfies $\delta < 2^{61}$ ($\approx 2.3\times10^{18}$);
a query spanning more than $10^{18}$ consecutive keys is a corpus scan, not a window
query.

**Semantic meaning of $\delta$.** Setting $\delta = 4^{32-j} - 1$ retrieves exactly
all documents whose transition sequences agree with the query in the first $j$
transitions (Corollary 3.7):

| Prefix length $j$ | Half-width $\delta$ | Semantic meaning |
|:-----------------:|:-------------------:|:-----------------|
| 32 | 0 | Exact key match |
| 31 | 3 | First 31 transitions match |
| 30 | 15 | First 30 transitions match |
| 28 | 255 | First 28 transitions match |
| 24 | 65 535 | First 24 transitions match |
| 16 | $\approx 4.3 \times 10^9$ | First 16 transitions match |

---

### 3.6 Bucket density

Documents with identical transition sequences share the same key and occupy the same
bucket. This is the intended behaviour: co-located documents traversed the same
semantic path.

Let $D$ be the number of distinct transition sequences in a corpus of $C$ documents.
Mean bucket size is $C/D$.

**Address space sparsity.** Even if every document in human history
($\approx 10^{10}$ documents) produced a unique 32-step trajectory, the occupancy of
the 64-bit key space is:

$$\frac{10^{10}}{2^{64}} \approx 5 \times 10^{-10}$$

The space is sparse; windows land in a sparse sea and return small, coherent
neighbourhoods. The oft-cited claim that $2^{64}$ addresses every atom in the
observable universe overstates it by 16 orders of magnitude — the observable universe
contains $\approx 10^{80}$ atoms and $2^{64} \approx 10^{19}$ — but the sparsity
conclusion is correct regardless.

**Corollary 3.9** *(Non-trivial windows).* For any practical corpus, a window of
half-width $\delta = 4^{32-j}$ with $j \leq 28$ returns a non-empty result only when
the query has neighbours that shared its first $j$ transitions. An empty result is
informative: no indexed document followed the same high-level semantic path as the
query.

---

[^1]: The $1/n_s$ variance normalisation ensures the dot product of two random unit
vectors has bounded variance regardless of dimension. This is the same normalisation
used in transformer attention ($1/\sqrt{d_k}$).
