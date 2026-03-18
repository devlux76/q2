# Design: Quaternary Semantic Quantization

## The Embedding Problem

### 0.0 What an Embedding Is

A transformer model maps a document — a sequence of tokens — to a vector in
$\mathbb{R}^n$ by passing it through the network and reading off activations, typically
mean-pooled over token positions and then L2-normalised. The result is a point on the
unit hypersphere:

$$e \in S^{n-1} = \left\{ x \in \mathbb{R}^n : \|x\| = 1 \right\}$$

Each coordinate $e_i$ is a real number constrained by $\sum_i e_i^2 = 1$. The natural
distance measure on this space is cosine similarity — equivalently, the dot product
between normalised vectors:

$$\text{sim}(u, v) = u \cdot v = \cos\theta_{uv}$$

where $\theta_{uv}$ is the angle subtended at the origin. Two documents are semantically
similar if and only if their embeddings are close on $S^{n-1}$; dissimilar if their
embeddings are far apart.

This geometric picture is complete and clean. The problem begins when you ask: similar
*according to whom*?

---

### 0.1 The Incommensurability Problem

The sphere $S^{n-1}$ has no preferred orientation. Every rotation $Q \in O(n)$ is an
isometry of the sphere: it maps $S^{n-1}$ to itself and preserves all pairwise angles.
The dot product between any two points is unchanged:

$$u \cdot v = (Qu) \cdot (Qv)$$

This is not a technical nuance; it is a structural obstruction. A model trained on a
corpus learns to place semantically similar documents near each other on the sphere, but
the *absolute position* of any concept — which hemisphere it inhabits, which axis it is
close to — is determined entirely by the random initialisation and the order in which
the training data arrived. A second model trained on the same corpus will produce an
embedding space related to the first by some unknown rotation $Q \in O(n)$:

$$e_B(x) \approx Q \cdot e_A(x)$$

where $Q$ encodes every accident of initialisation and training order. $Q$ is not
available. There is no way to recover it from the models' weights, and it cannot be
computed without a large set of paired examples and a Procrustes alignment procedure.

**The consequence.** For any two documents $x, y$ embedded by different models:

$$e_A(x) \cdot e_B(y) = e_A(x)^T e_B(y)$$

is a number whose sign and magnitude depend entirely on $Q$, which is arbitrary. The
dot product is not a measure of semantic similarity; it is a measurement of how a
random rotation happens to align two points from two unrelated coordinate systems. It
is, for all retrieval purposes, noise.

This is the incommensurability of embeddings: two embedding spaces that are both
internally coherent and semantically organised are mutually unintelligible without
explicit alignment. An index built from embeddings of one model cannot be queried with
embeddings from another.

**Incommensurability also applies to untrained models.** An untrained transformer
projects documents onto $\mathbb{R}^n$ using random weight matrices. The resulting
vectors cluster near the equator of $S^{n-1}$ by the central limit theorem — each
coordinate is a sum of many random terms — but the geometry encodes no semantic
content. There is no training signal that has pushed similar documents together, so
similar and dissimilar documents are geometrically indistinguishable. The activations
are noise, uniformly distributed on $S^{n-1}$ from the retrieval system's perspective.
An embedding is only meaningful as a retrieval key after the model has been trained to
make it so.

---

### 0.2 The Hypersphere: Curse of Dimensionality

The space on which embeddings live — $S^{n-1}$ — behaves increasingly pathologically
as $n$ grows. Understanding this geometry is prerequisite to understanding why the
quantization scheme in the following sections is the right answer.

**Volume of the $n$-ball.** The $n$-dimensional ball of radius $R$:

$$V_n(R) = \frac{\pi^{n/2}}{\Gamma\!\left(\tfrac{n}{2}+1\right)} R^n$$

For large $n$, $\Gamma(n/2+1) \approx \sqrt{\pi n}\,(n/2e)^{n/2}$ by Stirling, so:

$$V_n(1) \sim \frac{1}{\sqrt{\pi n}}\left(\frac{2\pi e}{n}\right)^{n/2} \to 0 \quad \text{as } n \to \infty$$

The unit ball *collapses in volume* as dimension grows. Simultaneously the surface area
$\mathcal{A}_{n-1}(R) = n V_n(R) / R$ also collapses. The sphere becomes an
infinitesimally thin skin around nothing.

**Shell concentration.** The fraction of the ball's volume contained in the outer shell
of thickness $\epsilon$ is:

$$f_{\text{shell}}(n,\,\epsilon) = \frac{V_n(R) - V_n(R-\epsilon)}{V_n(R)}
= 1 - \left(1 - \frac{\epsilon}{R}\right)^n$$

For the unit ball ($R = 1$):

$$f_{\text{shell}}(n,\,\epsilon) = 1 - (1-\epsilon)^n$$

As $n \to \infty$ for any fixed $\epsilon > 0$:

$$f_{\text{shell}}(n,\,\epsilon) \to 1$$

Every scrap of volume migrates to the surface. For small $\epsilon$ the exponential
form is revealing:

$$f_{\text{shell}}(n,\,\epsilon) \approx 1 - e^{-n\epsilon}$$

The shell thickness $\epsilon^*$ required to capture fraction $f$ of the total volume is:

$$\epsilon^*(f,\,n) = 1 - (1-f)^{1/n} \approx \frac{-\ln(1-f)}{n}$$

| Fraction captured | Shell thickness |
|-------------------|----------------|
| 63.2% | $1/n$ |
| 86.5% | $2/n$ |
| 95.0% | $3/n$ |
| 99.0% | $4.605/n$ |
| 99.9% | $6.908/n$ |

The shell containing 63% of the volume has radial thickness exactly $1/n$. The shell
containing 99% has thickness $\approx 4.6/n$. In 256 dimensions, 99% of the ball's
volume lies within a shell of radial thickness $4.6/256 \approx 0.018$ — less than 2%
of the radius.

**Random vectors are nearly orthogonal.** For $u, v \sim \text{Uniform}(S^{n-1})$:

$$\mathbb{E}[u \cdot v] = 0 \qquad \text{Var}(u \cdot v) = \frac{1}{n}$$

The standard deviation of the dot product between two random unit vectors is
$1/\sqrt{n}$. For $n = 256$ this is $\approx 0.063$. The entire range of cosine
similarity $[-1, +1]$ is compressed into fluctuations of order $\pm 0.06$. A trained
model must place semantically similar documents closer than this noise floor; without
training, no signal is detectable.

Each coordinate of a random unit vector satisfies:

$$\mathbb{E}[e_i] = 0 \qquad \mathbb{E}[e_i^2] = \frac{1}{n} \qquad \text{Var}(e_i^2) = \frac{2}{n^2(n+2)}$$

The expected squared magnitude of each coordinate is $1/n$, and the fluctuations
around this are $O(n^{-3/2})$ — tightly concentrated. In high dimensions, the
coordinates of a random unit vector are essentially i.i.d. $\mathcal{N}(0, 1/n)$.

---

### 0.3 The Planck Limit and the Holographic Analogy

The shell thickness $\epsilon^*(n) \sim c/n$ (where $c = O(1)$ sets the captured
fraction) shrinks without bound as dimension grows. This is a purely geometric
statement. But it has a physical corollary that illuminates why embedding dimension
cannot simply be made arbitrarily large.

**The crust thickness in absolute units.** Suppose the semantic space has a
characteristic physical scale $L$ — the radius of the meaningful region of $S^{n-1}$
in some ambient metric. The absolute thickness of the shell is:

$$\delta(n) = \frac{c \cdot L}{n}$$

There is a physical floor below which no distinction is resolvable: the Planck length,

$$\ell_P = \sqrt{\frac{\hbar G}{c^3}} \approx 1.616 \times 10^{-35} \;\text{m}$$

Below $\ell_P$, the concepts of distance and position lose operational meaning.
Setting $\delta(n_{\max}) = \ell_P$:

$$n_{\max} = \frac{c \cdot L}{\ell_P}$$

Beyond $n_{\max}$ dimensions, the radial distinctions implied by additional coordinates
are finer than Planck-scale resolution. Adding dimensions does not add information;
it subdivides already-indistinguishable states.

**The holographic analogy.** The Bekenstein-Hawking bound states that the maximum
entropy (information content) of any physical system enclosed by a surface of area $A$
is bounded by:

$$S_{\max} = \frac{A}{4\,\ell_P^2}$$

Information is bounded by *surface area*, not volume. The same structure emerges
geometrically from the hypersphere as $n$ grows:

1. All volume concentrates at the surface: $f_{\text{shell}} \to 1$.
2. The interior contributes negligible measure.
3. All distinguishable states live on $S^{n-1}$, not in the ball.
4. The capacity of the space is determined by how finely $S^{n-1}$ can be
   partitioned, which is a surface quantity — not a volume quantity.

The surface area of $S^{n-1}$ is:

$$\mathcal{A}_{n-1}(1) = \frac{2\pi^{n/2}}{\Gamma(n/2)}$$

This grows with $n$, but the minimum distinguishable angular separation between two
points also grows (as $\sim 1/\sqrt{n}$ from the concentration result above). The
effective number of distinguishable directions — the packing number of $S^{n-1}$ at
angular resolution $\Delta\theta$ — grows, but the *marginal* information per additional
dimension decreases as the already-thin shell continues to compress.

The holographic analogy is this: the semantic content of a document is encoded in
*which region of the sphere surface* its embedding occupies, not in its distance from
the origin. The sphere is already the hologram of the document. The quantization
problem — which is the subject of the remainder of this document — is the problem of
discretizing that hologram efficiently.

---

### 0.4 The Correct Metric: L1 and the Taxicab Unit Ball

The L2 distance between two points couples all $n$ coordinates under a square root:

$$d_2(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$

This coupling is the source of the shell-concentration pathology above, requires
$\pi^{n/2}$ in the volume formula, and makes the distance computation non-separable.

The L1 metric decouples the coordinates:

$$d_1(u, v) = \sum_{i=1}^n |u_i - v_i|$$

The L1 unit ball is the cross-polytope:

$$B_1^n = \left\{ x \in \mathbb{R}^n : \sum_{i=1}^n |x_i| \leq 1 \right\}$$

In two dimensions this is the diamond with vertices at $(\pm 1, 0)$ and $(0, \pm 1)$.
Its L1 boundary has four edges, each of L1 length 1, for a total L1 "circumference" of
8 over a diameter of 2:

$$\pi_1 = \frac{8}{2} = 4$$

$\pi$ is exactly 4 in the taxicab metric. No transcendental. The circle is a diamond.
The hypersphere is a cross-polytope.

The axis-aligned Cartesian grid is the natural grid for the L1 ball: the diamond's
faces are the coordinate halfspaces. There is no mismatch between the ball and the
grid. The squaring is exact.

The $n$-dimensional cross-polytope has $2n$ vertices at $\pm e_i$ for each standard
basis vector, and $2^n$ facets. Points on its boundary satisfy:

$$\sum_{i=1}^n |x_i| = 1$$

Each coordinate is an independent 1D displacement. The total L1 distance is their sum.

---

### 0.5 The Quaternary Representation Is the Coordinate

In L1 geometry, $d_1(u, v)$ reduces to $n$ independent scalar problems: for each
dimension $i$, how far apart are $u_i$ and $v_i$? The full distance is their sum, with
no coupling.

A single coordinate $x_i \in \mathbb{R}$ has two structural features relevant to
position on the L1 unit ball: its sign (which side of the origin) and its magnitude
class (near the origin or far from it). These two binary decisions produce four cells.
Fewer cells collapse either sign or magnitude class. More cells subdivide within a
cell, encoding intra-cell position that does not contribute to the L1 distance between
cells. Four is the minimum resolution that preserves both features.

The four cells $\{A, B, C, D\}$ correspond to $\{0, 1, 2, 3\}$ in $\mathbb{Z}_4$.
The Lee metric on $\mathbb{Z}_4$:

$$d_L(u_i, v_i) = \min(|u_i - v_i|,\; 4 - |u_i - v_i|)$$

is the L1 distance on the 4-point cycle. The cyclic wrap — $d_L(D, A) = 1$ — reflects
that strong-negative and strong-positive are both extreme points of the same axis, and
in the L1 cross-polytope they are adjacent vertices connected by an edge.

Extended to vectors:

$$d_L(u, v) = \sum_{i=1}^n \min(|u_i - v_i|,\; 4 - |u_i - v_i|)$$

This is the exact L1 distance on $\mathbb{Z}_4^n$. The Gray map (Section 0.7) makes
it computable by `popcnt(XOR)` on 64-byte vectors without decoding.

**The north-pole observation.** At the vertex $(D, D, \ldots, D)$ of the
cross-polytope — all coordinates at their maximum — every perturbation reduces at
least one coordinate. There is no rotational ambiguity, no choice of which arc to
follow, no $\pi$ to integrate over. The ordering $A < B < C < D$ on each axis is the
complete geometric structure of that dimension. At the poles of the cross-polytope,
every direction is toward the interior. The geometry is purely ordinal.

The float32 activation value is not the ground truth that the quaternary symbol
approximates. It is an overcomplete representation of an ordinal position. The bits
encoding intra-cell displacement contribute to the L2 norm and to the shell
concentration of Section 0.2 — they are not recoverable signal for L1 retrieval. The
quantization discards them correctly.

---

### 0.6 Implications for Retrieval System Design

Three conclusions follow.

**Incommensurability is absolute.** The quantization thresholds are calibrated from
one model's activation distribution. A different model produces different thresholds.
The quaternary codes are not comparable across models. An index must be rebuilt when
the model changes.

**The thermal constraint and the geometric constraint coincide.** Cross-model
alignment requires a Procrustes solve over paired examples — geometrically necessary
and thermally impossible on constrained hardware. Using the LLM's own activations
satisfies both constraints simultaneously.

**The quantization problem is now stated correctly.** The question is not: how many
bits per dimension approximate angular distance on $S^{n-1}$? The question is: what is
the natural discrete coordinate system of the L1 unit ball? The answer is four cells
per dimension — the unique minimum preserving sign and magnitude class — derived in
Section 1.5.

---

## The Quantization Problem

### 1.1 Motivation: The On-Device Thermal Constraint

Running a large language model on a consumer device — a phone, a laptop, a tablet —
already operates near the thermal ceiling of that hardware. Inference requires sustained use
of every available vector execution unit. A second dedicated embedding model, run in
parallel or in sequence to produce semantic index vectors, is not a cost that can be paid in
this environment. The device will throttle, the battery will drain, and the user will notice.

The question this paper addresses is therefore: given that an LLM is already running and
already producing activations, what is the highest-quality semantic index that can be
constructed from those activations alone, with no additional model, at a fixed and
predictable compute budget?

The answer is a quaternary quantization scheme. The remainder of this section derives it
from first principles, treating the quality of competing schemes — binary, ternary, and
standard Q4 — as a progression that makes the choice inevitable rather than arbitrary.

---

### 1.2 Binary Quantization and Its Structural Deficit

The simplest possible compression of a float32 activation $v_i \in \mathbb{R}$ is the sign bit:

$$q_{\text{bin}}(v_i) = \begin{cases} 0 & \text{if } v_i \leq 0 \\ 1 & \text{if } v_i > 0 \end{cases}$$

This produces one bit per dimension. For a 256-dimensional embedding, the result is a
256-bit vector. The natural distance measure on this representation is Hamming distance:
the number of positions where two vectors disagree.

**What binary quantization loses.** The float32 value encodes both direction (sign) and
magnitude (distance from zero). Binary quantization discards magnitude entirely. Two
activations $v_i = 0.01$ and $v_i = 4.7$ both map to 1. Under the original float32 metric
they are far apart; under binary Hamming distance they are identical.

More precisely, consider two document vectors $u, v \in \mathbb{R}^n$ and their binary
quantizations $\hat{u}, \hat{v} \in \{0,1\}^n$. The Hamming distance $d_H(\hat{u}, \hat{v})$
estimates the angle between $u$ and $v$ but carries no information about the magnitude
of agreement. A near-miss (small positive vs. small positive) and a strong agreement (large
positive vs. large positive) are indistinguishable from a strong disagreement (large positive
vs. large negative) when the magnitude structure is collapsed.

**The information loss is quantifiable.** For activations drawn from $\mathcal{N}(0, \sigma^2)$,
the mutual information between $v_i$ and $q_{\text{bin}}(v_i)$ is:

$$I(v_i;\, q_{\text{bin}}(v_i)) = H(q_{\text{bin}}(v_i)) = 1 \text{ bit}$$

The original float32 value carries $\log_2(2^{32}) = 32$ bits of representation capacity.
Binary quantization retains at most 1 bit of that signal per dimension. The semantic content
encoded in magnitude — which for mean-pooled transformer activations is substantial — is
structurally absent from the index.

---

### 1.3 Ternary Quantization and Its Remaining Deficit

Ternary quantization adds a central state:

$$q_{\text{tern}}(v_i) = \begin{cases} - & \text{if } v_i \leq -\tau \\ 0 & \text{if } -\tau < v_i \leq \tau \\ + & \text{if } v_i > \tau \end{cases}$$

for some threshold $\tau > 0$. This recovers one additional bit of information: whether the
activation is near the decision boundary or away from it. A value of 0 is semantically
distinct from a strong positive or strong negative. Centrality is now representable.

**What ternary quantization still loses.** The ternary scheme is symmetric around zero.
Given a value in the positive-strong class, there is no encoding of whether that value arrived
at its current position from a small positive or was always strongly activated. More
fundamentally for retrieval: the ternary alphabet contains no representation of the
complementary relationship between dimensions. Strong negative and strong positive are
maximally distant from each other, but there is no algebraic structure that makes this
explicit. The distance between $+$ and $-$ is 2 hops through $0$; the distance between
$+$ and $+$ is 0. But the group structure is $\mathbb{Z}_3$, not a ring with a natural
complement involution.

This matters because semantic space is not $\mathbb{Z}_3$-structured. Antonyms,
negations, and causal inversions are not simply "far away" from a concept — they bear a
specific complementary relationship to it. Ternary quantization has no way to encode or
exploit that relationship.

**The remaining information deficit.** For activations drawn from $\mathcal{N}(0, \sigma^2)$,
with $\tau$ chosen to make the three states equiprobable:

$$I(v_i;\, q_{\text{tern}}(v_i)) = \log_2 3 \approx 1.585 \text{ bits}$$

Ternary recovers 0.585 bits over binary. The complement structure — the relationship
between a concept and its semantic opposite — remains entirely unrepresented.

---

### 1.4 Standard Q4 and Why It Is a Different Problem

Standard 4-bit quantization as applied in LLM weight compression (GPTQ, AWQ, and related
methods) assigns 4 bits per parameter using a learned or analytically derived codebook that
minimises reconstruction error for that parameter's distribution. It is optimised for the
objective:

$$\min_{\hat{W}} \| W - \hat{W} \|_F^2$$

subject to $\hat{W}$ having 4-bit entries, encoding $2^4 = 16$ distinct levels per dimension.

This is a different problem from semantic quantization of activation vectors for retrieval.
Weight quantization optimises for numerical fidelity of a weight matrix. Activation
quantization for retrieval optimises for preservation of semantic distance between
document representations. The metric, the objective, and the distribution being
compressed are all different.

Q4 weight quantization is not a competitor to the scheme presented here. It is a solution to
a different problem operating in a different part of the inference pipeline. Comparisons are
meaningless without this distinction.

---

### 1.5 The Quaternary Semantic Quantization Scheme

The preceding analysis identifies what a correct scheme must provide:

1. **Sign** — which side of the semantic hyperplane the activation falls on.
2. **Magnitude class** — whether the activation is near the boundary or strongly committed.
3. **Complement structure** — an algebraic relationship between a symbol and its semantic
   opposite that is preserved under the distance metric.
4. **Minimum alphabet size** — the fewest symbols that satisfy all three requirements
   simultaneously.

Requirements 1 and 2 together demand at least four states: two on each side of zero, one
near and one far. Requirement 3 demands that the algebra on these four states admits a
complement involution — a map $\theta$ satisfying $\theta^2 = \text{id}$ and $\theta(x) \neq x$
for all $x$.

Requirement 4 asks whether three states suffice. They do not: with three states $\{-, 0, +\}$,
the only fixed-point-free involution is $\theta(-) = +$, $\theta(+) = -$, $\theta(0) = \;?$.
There is no element for 0 to map to that satisfies the involution without a fixed point. Four is
the minimum.

**The four states.** Label the four states by signed magnitude class:

$$\{A,\; B,\; C,\; D\} \quad \longleftrightarrow \quad \{\text{strong-},\; \text{weak-},\; \text{weak+},\; \text{strong+}\}$$

**The quantization function.** For an activation $v_i$ drawn from $\mathcal{N}(0, 1/n_s)$,
where $n_s$ is the embedding dimension,[^1] the maximum-entropy condition requires all
four states to be equiprobable:

$$P(v_i \leq -\tau^*) = P(-\tau^* < v_i \leq 0) = P(0 < v_i \leq \tau^*) = P(v_i > \tau^*) = \tfrac{1}{4}$$

By symmetry $\tau_1 = 0$ is fixed. The single non-trivial threshold satisfies:

$$\tau^* = \frac{\Phi^{-1}(3/4)}{\sqrt{n_s}} \approx \frac{0.674}{\sqrt{n_s}}$$

The quantization function is then:

$$q(v_i) = \begin{cases} A & \text{if } v_i \leq -\tau^* \quad (\text{strong negative}) \\ B & \text{if } -\tau^* < v_i \leq 0 \quad (\text{weak negative}) \\ C & \text{if } 0 < v_i \leq \tau^* \quad (\text{weak positive}) \\ D & \text{if } v_i > \tau^* \quad (\text{strong positive}) \end{cases}$$

**Empirical threshold calibration.** The theoretical threshold assumes Gaussian activations.
In practice, the threshold is estimated from a reservoir sample of 1024 document activations
per compaction cycle, using the empirical 25th and 75th percentiles of $v_i$ (not $|v_i|$) to
place the boundary between weak and strong states on each side of zero symmetrically.
This keeps the symbol distribution close to equiprobable without assuming any specific
activation distribution. See Section 3 for the calibration procedure.

**Maximum entropy achieved.** Under the empirical calibration target, each dimension
independently carries:

$$I(v_i;\, q(v_i)) = \log_2 4 = 2 \text{ bits}$$

This is the maximum achievable for a 4-symbol alphabet. All four states are equiprobable;
no state is wasted on rare events.

| Pair type | Example | Lee distance |
|-----------|---------|--------------|
| Identical | $A$–$A$ | 0 |
| Adjacent | $A$–$B$, $B$–$C$, $C$–$D$ | 1 |
| Cyclic wrap | $D$–$A$ | 1 |
| Complement | $A$–$C$, $B$–$D$ | 2 |

---

### 1.6 The Lee Metric: Derived, Not Chosen

Given four ordered states $\{A=0,\; B=1,\; C=2,\; D=3\}$ arranged cyclically in $\mathbb{Z}_4$,
the natural distance is one that respects the ordering: adjacent states should cost less than
non-adjacent states.

**Definition.** The Lee distance on $\mathbb{Z}_4$ is:

$$d_L(u, v) = \min(|u - v|,\; 4 - |u - v|)$$

This assigns the distances shown in the table above.

**Why this is the correct metric for semantic retrieval.** A weak-negative and a weak-positive
activation ($B$–$C$) differ by one step in magnitude class. They are semantically similar: a
concept was weakly present in both documents, differing only in sign. Lee distance 1 reflects
this. The complement pairs $A$–$C$ and $B$–$D$ encode semantic opposition: one document
activates a dimension strongly in one direction while the other activates it in the
complementary direction. Lee distance 2 reflects this. Note that strong-negative and
strong-positive ($A$–$D$) are adjacent in the cyclic order (distance 1 via the $D \to A$ wrap),
not complements — they share strong commitment to their respective directions, which the
cyclic metric correctly treats as a single step. Hamming distance on raw 2-bit strings does
not capture any of these distinctions without the cyclic structure.

**Toy example.** Consider two 4-dimensional vectors:

$$u = [A, B, C, D] = [0, 1, 2, 3]$$
$$v = [A, C, C, B] = [0, 2, 2, 1]$$

Position-wise Lee distances: $d_L(0,0)=0$, $d_L(1,2)=1$, $d_L(2,2)=0$,
$d_L(3,1)=\min(2,2)=2$. Total: $d_L(u,v) = 3$.

**Extending to vectors.** For vectors $u, v \in \mathbb{Z}_4^n$, the Lee distance is:

$$d_L(u, v) = \sum_{i=1}^{n} \min(|u_i - v_i|,\; 4 - |u_i - v_i|)$$

For $n = 256$ dimensions, the maximum Lee distance is $256 \times 2 = 512$.

---

### 1.7 The Gray Map: The Unique Binary Encoding That Preserves Lee Distance

Given the Lee metric on $\mathbb{Z}_4$, we require a binary encoding $\phi: \mathbb{Z}_4 \to \{0,1\}^2$
such that the Hamming distance on the encoded vectors equals the Lee distance on the
original symbols. This makes SIMD `popcnt`-on-XOR — the fastest available hardware
operation for distance computation — compute the exact Lee distance without any
symbol-level decoding.

**Theorem 0.1** *(Hammons, Kumar, Calderbank, Sloane, Solé, 1994 [1]).* The Gray map
$\phi: \mathbb{Z}_4 \to \{0,1\}^2$ defined by:

$$\phi(0) = 00 \qquad \phi(1) = 01 \qquad \phi(2) = 11 \qquad \phi(3) = 10$$

is an isometry from $(\mathbb{Z}_4^n, d_L)$ to $(\{0,1\}^{2n}, d_H)$:

$$d_H(\phi(u), \phi(v)) = d_L(u, v) \quad \text{for all } u, v \in \mathbb{Z}_4^n$$

**Lemma 0.2** *(Single-symbol verification).* For any $u, v \in \mathbb{Z}_4$:

$$d_H(\phi(u), \phi(v)) = \min(|u-v|,\; 4-|u-v|)$$

*Proof.* Enumerate all ten unordered pairs:

| Pair | $\phi$-encodings | XOR | Hamming | Lee |
|------|-----------------|-----|---------|-----|
| $(0,0),(1,1),(2,2),(3,3)$ | equal | $00$ | 0 | 0 |
| $(0,1),(1,2),(2,3)$ | adjacent | $01$ or $10$ | 1 | 1 |
| $(0,3)$ | cyclic wrap | $10$ | 1 | $\min(3,1)=1$ |
| $(0,2),(1,3)$ | complement | $11$ | 2 | $\min(2,2)=2$ |

Hamming equals Lee in every case. $\square$

**Consequence.** `popcnt(XOR)` applied to Gray-encoded vectors computes cyclic Lee
distance on $\mathbb{Z}_4^n$ without decoding.

The Gray map is the unique such encoding up to relabelling. Any binary encoding of
$\mathbb{Z}_4$ that preserves Lee distance under Hamming must map the cyclic structure
of $\mathbb{Z}_4$ to a reflected binary code. The 2-bit reflected binary Gray code is that
structure. The choice of encoding is therefore not a design decision but a consequence of
the metric requirement.

**Observation: Gray encoding admits a closed-form formula.** For any $n \in \{0,1,2,3\}$:

$$\phi(n) = n \oplus (n \gg 1)$$

Verified: $\phi(0)=0\oplus 0=00$, $\phi(1)=1\oplus 0=01$, $\phi(2)=2\oplus 1=3=11$,
$\phi(3)=3\oplus 1=2=10$. This identity is used directly in the SIMD implementation
(Section 1.9) to compute Gray codes for four symbols simultaneously in a single vector XOR
instruction.

**The encoding table.** Combining the quantization function with the Gray map:

| Symbol | Semantic meaning | $\mathbb{Z}_4$ value | $\phi$-encoding |
|--------|-----------------|---------------------|-----------------|
| $A$ | Strong negative | 0 | `00` |
| $B$ | Weak negative | 1 | `01` |
| $C$ | Weak positive | 2 | `11` |
| $D$ | Strong positive | 3 | `10` |

A 256-dimensional vector encodes to $256 \times 2 = 512$ bits = 64 bytes.

---

### 1.8 The Complement Involution

The fixed-point-free complement involution required in Section 0.5 is explicit in the Gray
map. Define $\theta: \mathbb{Z}_4 \to \mathbb{Z}_4$ by:

$$\theta(x) = x + 2 \pmod{4}$$

This gives $\theta(A)=C$, $\theta(B)=D$, $\theta(C)=A$, $\theta(D)=B$. In the $\phi$-encoding,
$\theta$ is bitwise NOT on 2-bit strings: $\phi(\theta(x)) = \overline{\phi(x)}$.

**Proposition 0.3** *(Universal complement identity).* For any vector $v \in \mathbb{Z}_4^{256}$
and its complement $\bar{v}$ (defined componentwise by $\theta$):

$$v \oplus \bar{v} = \texttt{MaxUINT512} = 2^{512} - 1$$

*Proof.* For any 2-bit Gray encoding $e$, $e \oplus \overline{e} = 11$ by definition of bitwise
NOT. All 256 symbol positions contribute $11$. Concatenated over 512 bits:
$\texttt{MaxUINT512}$. $\square$

**Corollary** *(Self-complement exclusion).* No vector $v$ satisfies $v = \bar{v}$.

*Proof.* If $v = \bar{v}$ then $v \oplus \bar{v} = 0 \neq \texttt{MaxUINT512}$. Contradiction. $\square$

---

[^1]: The $1/n_s$ variance normalisation ensures that the dot product of two random unit
vectors has bounded variance regardless of dimension. This is the same normalisation used
in transformer attention ($1/\sqrt{d_k}$) and is standard for high-dimensional embeddings.
