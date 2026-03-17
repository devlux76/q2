# Design: Quaternary Semantic Quantization

## The Quantization Problem

### 0.1 Motivation: The On-Device Thermal Constraint

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

### 0.2 Binary Quantization and Its Structural Deficit

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

### 0.3 Ternary Quantization and Its Remaining Deficit

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

### 0.4 Standard Q4 and Why It Is a Different Problem

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

### 0.5 The Quaternary Semantic Quantization Scheme

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

### 0.6 The Lee Metric: Derived, Not Chosen

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

### 0.7 The Gray Map: The Unique Binary Encoding That Preserves Lee Distance

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
(Section 0.9) to compute Gray codes for four symbols simultaneously in a single vector XOR
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

### 0.8 The Complement Involution

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
