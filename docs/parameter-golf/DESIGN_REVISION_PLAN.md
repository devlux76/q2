# DESIGN.md Revision Plan

> Section-by-section assessment: what is removed, revised, new, or unchanged
> when Q² generalizes beyond semantic embeddings and integrates Wildberger-Rubine.

---

## Structural change

The current document has three parts:

1. **The Embedding Problem** (§1) — motivation from embeddings
2. **Quantization** (§2) — the Z₄ machinery
3. **The Transition Key** (§3) — indexing

The revised document should have four parts:

1. **The Quantization Problem** — general motivation (what quantization is,
   why 4 cells, why L1). Embeddings become one application, not the frame.
2. **The Quaternary Coordinate** — the Z₄ machinery (Lee, Gray, complement).
   Unchanged in substance.
3. **The Transition Key** — indexing. Mostly unchanged, gains hyper-Catalan
   counting.
4. **The Combinatorial Structure** — NEW. Wildberger-Rubine integration:
   Geode factorization, Euler constraint, threshold geometry, mixed-precision.

---

## Section-by-section

### Title

| Current | Revised |
|:--------|:--------|
| "Quaternary Quantization: Design" | **Same** |

No change needed — it's already general.

---

### §1 header

| Current | Revised |
|:--------|:--------|
| "The Embedding Problem" | **"The Quantization Problem"** |

**REVISED.** The section should open with the general question — what is the
natural discrete coordinate system for a continuous signal? — not with
embeddings specifically.

---

### §1.1 Embeddings and the unit hypersphere

**REVISED → demoted.** The content is correct but application-specific. It
should move to a later section or appendix ("Application: Embeddings") or
become a motivating example within a broader §1.1 that introduces the general
setting: you have a point in $\mathbb{R}^n$ (or on $S^{n-1}$) and want to
discretize it.

The mean-pooling digression ("Why not mean-pool?") is entirely
embedding-specific. **Demote or remove** — it's a design note for the embedding
application, not a general principle.

---

### §1.2 Incommensurability

**REVISED → demoted.** Incommensurability is about comparing two embedding
models. It's not a general quantization problem — it's a property of the
embedding application. The mathematical content ($Q \in O(n)$, rotational
invariance) is sound but belongs under "Application: Embeddings."

The closing paragraph about relational geometry surviving rotation is the
important general insight. **Extract** the principle: quantization should capture
relational structure (differences, trajectories) rather than absolute
coordinates. That principle is general — it applies to any signal, not just
embeddings.

---

### §1.3 Concentration of measure

**SAME.** This is already general. Volume concentration, shell thickness, the
$1/\sqrt{n}$ noise floor — all apply to any high-dimensional quantization
problem. No changes needed.

---

### §1.4 The surface-area information bound

**REVISED.** The content stays, but gains the Wildberger-Rubine connection:

- **Add:** The Euler polytope formula $V - E + F = 1$ governs the combinatorial
  capacity of any quantization lattice. The number of cells ($F$), boundaries
  ($E$), and vertices ($V$) are constrained — you cannot design them
  independently. This is the combinatorial analog of the Bekenstein-Hawking
  surface-area bound already cited. (Ref: Wildberger & Rubine 2025, where the
  hyper-Catalan coefficient $(E-1)!/((V-1)! \cdot \mathbf{m}!)$ is governed by
  exactly this relation.)

- **Revise** the sentence "The semantic content of a document is encoded in
  which region of $S^{n-1}$ its embedding occupies" → generalize to "The
  information content of a signal is encoded in which region of the space its
  representation occupies."

---

### §1.5 The L1 metric and the cross-polytope

**SAME.** Already general. L1 decomposition, cross-polytope geometry, no
coupling between dimensions — this is framework-level mathematics. No changes.

---

### §1.6 The quaternary coordinate

**SAME.** Already general. The derivation of four cells from sign × magnitude
class does not depend on embeddings. The final paragraph about float32
activations being "overcomplete representations of an ordinal position" could
be generalized slightly — replace "float32 activation" with "continuous
measurement" — but the logic is unchanged.

Minor wording revision: "for L1 retrieval" → "for L1 distance computation."

---

### §1.7 Index design consequences

**REVISED → split.**

The three consequences currently listed are:

1. Coordinate-frame incommensurability — **embedding-specific, demote**
2. Thermal + geometric constraints coincide — **embedding-specific, demote**
3. The quantization problem restated — **general, keep and promote**

Consequence 3 is the key statement: "What is the natural discrete coordinate
system of the L1 unit ball?" This should be the *opening* of §1, not a
consequence buried in §1.7.

---

### §1.8 Relational geometry and the lingua franca

**REVISED → demoted.** The king/queen analogy, the lingua franca argument, the
transition key as "indexing shape rather than position" — all embedding-specific
narrative. The underlying principle (quantize trajectories, not positions) is
general and should be extracted into the framework. The specific embedding
application (cross-model invariance, Procrustes avoidance) moves to
"Application: Embeddings."

---

### §2 header: Quantization

**SAME.** Already general.

---

### §2.1 The thermal constraint

**REVISED.** Currently frames the constraint as "an LLM is already running on a
phone." This is one instance of a general principle: **quantization operates
under a resource budget.** The thermal constraint is real and important, but the
general statement is: given a fixed compute/memory/energy budget, what is the
highest-fidelity discrete representation?

Keep the thermal constraint as a concrete example. Add the general framing.

---

### §2.2 Binary quantization

**SAME.** Already general. Sign-bit quantization, 1 bit per dimension, Hamming
distance. No embedding-specific content.

---

### §2.3 Ternary quantization

**SAME.** Already general. The $\mathbb{Z}_3$ no-involution argument is pure
algebra.

---

### §2.4 Standard Q4 (different problem)

**REVISED.** Currently says "this is weight quantization, not activation
quantization for retrieval." In a general framework, the distinction is between:

- **Reconstruction quantization** (GPTQ, AWQ): minimize $\|W - \hat{W}\|_F^2$.
  The goal is to approximate the original signal.
- **Structural quantization** (Q²): preserve relational/topological structure.
  The goal is to preserve distances, not values.

The section should frame this as two different quantization objectives, not as
"different problem, not our concern."

---

### §2.5 Quaternary semantic quantization

**REVISED.** Title change: drop "semantic."

| Current | Revised |
|:--------|:--------|
| "Quaternary semantic quantization" | **"Quaternary quantization"** |

The four requirements (sign, magnitude class, complement structure, minimum
alphabet) are already general. No changes to the derivation.

**Add (Wildberger-Rubine):** After the empirical calibration paragraph, add a
new paragraph on **analytical threshold computation for non-Gaussian sources:**

> For source distributions expressible as polynomial or mixture models, the
> equiprobable threshold $\tau^*$ can be computed analytically via the
> hyper-Catalan series (Wildberger & Rubine 2025). The threshold equation
> $F(\tau) = k/4$ for CDF $F$ becomes a polynomial in the distribution
> parameters, and the series $\alpha = \sum_\mathbf{m} C_\mathbf{m} \cdot
> t_2^{m_2} t_3^{m_3} \cdots$ converges without iteration. Truncation order
> trades precision for compute cost — a natural fit for the resource-constrained
> setting of §2.1.

This does not replace the empirical calibration — it adds a second path.

---

### §2.6 The Lee metric

**REVISED (minor).** The paragraph "Why this metric matches semantic distance"
uses "semantic distance" language throughout. Generalize:

- "the concept was weakly activated" → "the coordinate was weakly committed"
- "semantic opposition" → "structural opposition"
- "Strong-negative and strong-positive share strong commitment to their
  respective directions" — this is already general, keep as-is.

The math, examples, and diagrams are unchanged.

---

### §2.7 The Gray map

**SAME.** Pure algebra: the $\phi$ isometry, Hammons et al. 1994, popcnt(XOR).
Nothing embedding-specific. No changes.

---

### §2.8 The complement involution

**SAME.** Pure algebra. No changes.

---

### §3 header: The Transition Key

**SAME.** Already general.

---

### §3.1 Run-reduction

**REVISED (minor).** Two small wording changes:

- "a document that visits a semantic state" → "a signal that visits a
  quantization state"
- "The key records which states were visited" — already general, keep.

The algorithm, lemmas, and proofs are unchanged.

---

### §3.2 The 64-bit integer key

**SAME.** Pure number theory. Base-4 encoding, 64-bit fit, bit layout. No
embedding-specific content.

---

### §3.3 MSB alignment and prefix semantics

**REVISED (minor).** "Semantic depth" → "resolution depth" or just "depth."
The term "semantic" here means "meaningful," which is general, but it will read
as embedding-specific in context.

---

### §3.4 Block file organisation

**SAME.** Pure index design. Block count, hex ranges, binary search. No changes.

---

### §3.5 Window queries

**REVISED (minor).** The table column "Semantic meaning" → "Meaning" or
"Interpretation." The content ("first 31 transitions match") is already general.

---

### §3.6 Bucket density

**REVISED.** Keep all existing content. **Add** the hyper-Catalan trie counting:

> **Admissible sequence count.** The transition trie has a root of arity $q = 4$
> and all subsequent nodes of arity $q - 1 = 3$ (each successor must differ from
> its predecessor). The number of distinct transition sequences of length $k$ is:
>
> $$D(k) = q \cdot (q-1)^{k-1} = 4 \cdot 3^{k-1}$$
>
> For $k = 32$ (the key capacity), $D(32) = 4 \cdot 3^{31} \approx
> 2.17 \times 10^{15}$, occupying $\approx 1.2 \times 10^{-4}$ of the $2^{64}$
> address space.
>
> More generally, the number of distinct subtree patterns of mixed arity in the
> transition trie — relevant when branching factors vary under transition
> constraints — is counted by the hyper-Catalan number
> $C_\mathbf{m} = (E-1)!/((V-1)! \cdot \mathbf{m}!)$ where $V - E + F = 1$
> (Wildberger & Rubine 2025). This provides exact combinatorial formulas for
> non-uniform bucket density when the source distribution favors certain
> branching patterns over others.

---

### NEW §4: The Combinatorial Structure

**NEW section.** This is where the bulk of the Wildberger-Rubine integration
lives.

#### §4.1 The Geode factorization and hierarchical quantization

The factorization $S - 1 = S_1 \cdot G$ as the algebraic skeleton of
progressive/multi-resolution quantization. First level ($S_1$) = coarse cell.
Geode ($G$) = refinement within. Recursive structure gives coarse-to-fine for
free. Connects to §3.3 (prefix semantics) and §3.5 (window queries).

#### §4.2 Euler's polytope formula as a quantization constraint

$V - E + F = 1$ constrains the topology of admissible quantization lattices.
For $\mathbb{Z}_4$: $V=4, E=4, F=1$. For product $\mathbb{Z}_4^n$: the face
structure is computable. When extending to $q > 4$, this gives an a priori
enumeration of admissible lattice geometries.

#### §4.3 Mixed-precision quantization

The Bi-Tri (and higher) hyper-Catalan arrays count the number of distinct
codebooks with $m_2$ binary dimensions, $m_3$ ternary, $m_4$ quaternary. This
bounds the search space for adaptive bit allocation: spend 2 bits on
high-variance dimensions, 1 bit on low-variance ones.

#### §4.4 Threshold geometry

For non-Gaussian distributions (mixtures, heavy-tailed), the equiprobable
threshold equation $F(\tau) = k/q$ is a polynomial in the distribution
parameters. The hyper-Catalan series solves it combinatorially. Truncation
order maps to precision/cost tradeoff.

#### §4.5 Reconstruction and series reversion

If Q² ever needs a decode path (lossy compression, not just retrieval), the
optimal reconstruction point $\mathbb{E}[x \mid q(x) = s]$ requires inverting
the CDF within each cell. The Lagrange-inversion / hyper-Catalan series
provides this without numerical root-finding.

---

### NEW: Application section (or appendix)

**Application: Embeddings.** Collects the embedding-specific material currently
scattered through §1:

- §1.1 (hypersphere, mean-pooling)
- §1.2 (incommensurability, Procrustes)
- §1.7 (thermal + geometric coincidence, coordinate-frame constraint)
- §1.8 (relational geometry, king/queen, lingua franca)
- §2.1 thermal constraint as concrete instance

This is not deleted — it's relocated. The embedding application is the primary
use case and should be presented prominently, but as an application of the
general framework rather than as the frame itself.

---

## Summary table

| Section | Status | Nature of change |
|:--------|:------:|:-----------------|
| Title | **Same** | — |
| §1 header | **Revised** | "Embedding Problem" → "Quantization Problem" |
| §1.1 | **Demoted** | Moves to Application: Embeddings |
| §1.2 | **Demoted** | Moves to Application: Embeddings (principle extracted) |
| §1.3 | **Same** | Already general |
| §1.4 | **Revised** | Gains Euler V-E+F connection; wording generalized |
| §1.5 | **Same** | Already general |
| §1.6 | **Same** | Already general (minor wording) |
| §1.7 | **Split** | Item 3 promoted to §1 opener; items 1-2 demoted |
| §1.8 | **Demoted** | Moves to Application: Embeddings (principle extracted) |
| §2.1 | **Revised** | Generalized framing; thermal example kept |
| §2.2 | **Same** | Already general |
| §2.3 | **Same** | Already general |
| §2.4 | **Revised** | Reframed as reconstruction vs. structural quantization |
| §2.5 | **Revised** | Drop "semantic" from title; add analytical thresholds |
| §2.6 | **Revised** | Minor wording: "semantic distance" → "structural distance" |
| §2.7 | **Same** | Pure algebra |
| §2.8 | **Same** | Pure algebra |
| §3.1 | **Revised** | Minor wording: "semantic state" → "quantization state" |
| §3.2 | **Same** | Pure number theory |
| §3.3 | **Revised** | Minor: "semantic depth" → "resolution depth" |
| §3.4 | **Same** | Pure index design |
| §3.5 | **Revised** | Minor: column header generalized |
| §3.6 | **Revised** | Gains hyper-Catalan trie counting |
| §4 | **New** | Combinatorial Structure (Wildberger-Rubine) |
| §4.1 | **New** | Geode factorization → hierarchical quantization |
| §4.2 | **New** | Euler V-E+F as lattice constraint |
| §4.3 | **New** | Mixed-precision counting |
| §4.4 | **New** | Threshold geometry for non-Gaussian |
| §4.5 | **New** | Reconstruction via series reversion |
| App. | **New** | Application: Embeddings (relocated from §1) |

**Count:** 8 Same, 12 Revised, 6 New, 0 Removed.

Nothing is deleted. Content moves; framing changes; new theory is added.
