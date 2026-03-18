# Quaternary Semantic Quantization: Predictions

> **Related documents:** [DESIGN.md](DESIGN.md) · [TESTING.md](TESTING.md)

Section references of the form §D-x.y refer to [DESIGN.md](DESIGN.md).
Section references of the form §T-x refer to [TESTING.md](TESTING.md).

---

## Contents

- [P1 — The canonical CGAT mapping](#p1--the-canonical-cgat-mapping)
- [P2 — Codon palindromes and hairpin density](#p2--codon-palindromes-and-hairpin-density)
- [P3 — Complement bigram suppression (CpG analog)](#p3--complement-bigram-suppression-cpg-analog)
- [P4 — Transition/transversion asymmetry and weighted Lee distance](#p4--transitiontransversion-asymmetry-and-weighted-lee-distance)
- [P5 — Reverse complement retrieval of semantic antonyms](#p5--reverse-complement-retrieval-of-semantic-antonyms)
- [P6 — Two-stage search architecture (PAM/seed analog)](#p6--two-stage-search-architecture-pamseed-analog)
- [P7 — Document secondary structure](#p7--document-secondary-structure)
- [P8 — Codon degeneracy and synonymous substitutions](#p8--codon-degeneracy-and-synonymous-substitutions)
- [P9 — Alphabet optimality (Kerdock/Preparata)](#p9--alphabet-optimality-kerdockpreparata)
- [Summary table](#summary-table)

---

## P1 — The canonical CGAT mapping

The Gray map $\phi$ of §D-2.7 assigns 2-bit encodings to the four Z₄ symbols. The
unique assignment of DNA bases $\{G, A, C, T\}$ to $\{0, 1, 2, 3\}$ that makes
Watson–Crick complementarity coincide with the complement involution $\theta$
(§D-2.8) is:

| Z₄ | $\phi$ | DNA base | Ring structure | Functional group |
|:--:|:------:|:--------:|:--------------:|:----------------:|
| 0  | `00`   | G        | Purine (two-ring) | Keto |
| 1  | `01`   | A        | Purine (two-ring) | Amino |
| 2  | `11`   | C        | Pyrimidine (one-ring) | Amino |
| 3  | `10`   | T        | Pyrimidine (one-ring) | Keto |

**The mapping is unique.** Watson–Crick pairs are $\{G, C\}$ and $\{A, T\}$. The
involution $\theta$ pairs $\{0,2\}$ and $\{1,3\}$. Mapping $G \mapsto 0$,
$C \mapsto 2$, $A \mapsto 1$, $T \mapsto 3$ (or its complement relabelling) is the
only assignment satisfying $\theta(G) = C$ and $\theta(A) = T$.

**The Gray bits encode the two chemical classification axes.**

- **Bit 0** (MSB of $\phi$): $0$ for $\{G, A\}$, $1$ for $\{C, T\}$.
  This is the purine/pyrimidine distinction — whether the base has two fused rings or
  one.
- **Bit 1** (LSB of $\phi$): $0$ for $\{G, T\}$, $1$ for $\{A, C\}$.
  This is the keto/amino distinction — whether the base carries a carbonyl or an
  amino group at the relevant position.

These two axes are the standard chemical classification of nucleobases used
independently of the Watson–Crick model. The Gray code encodes them simultaneously,
with no design choices remaining.

**The Lee metric has a chemical interpretation.**

- **Lee distance 1 pairs** — one chemical property changes:
  - $G$–$A$ (both purines, differ in keto/amino): transition mutation
  - $C$–$T$ (both pyrimidines, differ in keto/amino): transition mutation
  - $G$–$T$ (both keto, differ in purine/pyrimidine): transversion type 1
  - $A$–$C$ (both amino, differ in purine/pyrimidine): transversion type 1
- **Lee distance 2 pairs** — both chemical properties change:
  - $G$–$C$: Watson–Crick complement pair
  - $A$–$T$: Watson–Crick complement pair

Lee distance 2 is exactly Watson–Crick complementarity. In molecular evolution, the
Lee-distance-2 substitutions (complement transversions) are the most disruptive
because they cross both chemical axes simultaneously.

**Consequence for retrieval.** The Lee metric on Z₄ is equivalent to Hamming distance
in 2D chemical property space $\{\text{purine/pyrimidine}\} \times \{\text{keto/amino}\}$.
The mapping from embedding space to Z₄ is therefore a projection onto a chemically
natural 2D coordinate system. This is not an analogy; it is the same algebraic
structure.

---

## P2 — Codon palindromes and hairpin density

A **codon** is any triplet of consecutive symbols in a transition sequence
$R = (r_0, r_1, \ldots, r_{k-1})$. Adjacent symbols in $R$ are always distinct
($r_i \neq r_{i+1}$ for all $i$, by construction of run-reduction).

**Definition.** A codon $(r_i,\ r_{i+1},\ r_i)$ is a *palindrome codon*. It
represents a trajectory that departs from state $r_i$, visits $r_{i+1}$, and returns.

The three palindrome types differ by the Lee distance of the intermediate step:

| Form | Example | Lee dist. | Character |
|:----:|:-------:|:---------:|:---------|
| Adjacent palindrome | $(A, B, A)$ | 1 | Narrow excursion; returns from nearest neighbour |
| Cyclic-wrap palindrome | $(A, D, A)$ | 1 | Returns from cyclic-adjacent extremum |
| **Complement palindrome** | $(A, C, A)$, $(B, D, B)$ | **2** | Visits semantic antipode and returns |

A complement palindrome is exactly $(x,\ \theta(x),\ x)$: the trajectory crosses to
the complement state and returns. In the Gray encoding, $\phi(\theta(x)) = \overline{\phi(x)}$,
so the intermediate symbol has all bits flipped. The biological correspondence is direct:
an RNA hairpin loop forms where a strand folds back on itself via Watson–Crick
complementarity; here the fold is in the transition sequence, and the complementarity
is $\theta$.

**Hairpin density.** Define:

$$\rho_{\text{hp}}(R) = \frac{\bigl|\{i : r_{i+1} = \theta(r_i)\ \text{and}\ r_{i+2} = r_i\}\bigr|}{|R| - 2}$$

for $|R| \geq 3$, and $0$ otherwise.

**Null baseline.** For a uniformly random transition sequence of length $|R|$,
conditioning on $r_i = x$ gives $r_{i+1}$ uniform over the three values
$\{A,B,C,D\} \setminus \{x\}$, so $P(r_{i+1} = \theta(r_i)) = 1/3$. Given
$r_{i+1} = \theta(r_i)$, $r_{i+2}$ is uniform over the three values excluding
$r_{i+1}$, exactly one of which equals $r_i$, so $P(r_{i+2} = r_i) = 1/3$:

$$\mathbb{E}[\rho_{\text{hp}}]_{\text{null}} = \frac{1}{3} \cdot \frac{1}{3} = \frac{1}{9} \approx 0.111$$

**Prediction.** A complement palindrome codon in a document's transition sequence
indicates a dimension that was visited in opposition and returned from: the embedding
crossed to the semantic complement of some direction before settling back. Documents
containing dialectical or contrastive language should produce transition sequences with
$\rho_{\text{hp}} > 1/9$; documents that commit to a semantic complement without
returning should produce $\rho_{\text{hp}} < 1/9$.

Three probe classes make this precise:

| Class | Example | Predicted $\rho_{\text{hp}}$ |
|:-----:|:-------:|:---:|
| Direct | "Optimism is warranted." | $\approx 1/9$ |
| Dialectical | "Optimism is warranted; one could argue for pessimism, yet on balance optimism prevails." | $> 1/9$ |
| Negated | "Optimism is not warranted." | $< 1/9$ |

The ordering prediction is:

$$\rho_{\text{hp}}(\text{Dialectical}) > \rho_{\text{hp}}(\text{Direct}) \approx \frac{1}{9} > \rho_{\text{hp}}(\text{Negated})$$

**Antonym retrieval corollary.** Two documents about semantically opposed concepts
(e.g. *optimism* and *pessimism*) both traverse the same semantic axis — one as
$(x,\ \theta(x),\ x)$ and the other as $(\theta(x),\ x,\ \theta(x))$. The set of
complement palindrome codons in their respective transition sequences should overlap
more than it does for two unrelated documents. Complement-palindrome overlap between
query and candidate is therefore a signal of semantic opposition, orthogonal to Lee
distance.

Test protocol: §T-2 (code corpus), §T-3 (embedding models).

---

## P3 — Complement bigram suppression (CpG analog)

In vertebrate genomes, the dinucleotide CG is severely underrepresented — typically
at 20–25% of its expected frequency under a base-composition model — because CpG sites
are hypermutable: methylation at cytosine followed by deamination converts CG to TG at
elevated rates, depleting CG over evolutionary time.

Under the CGAT mapping (P1), a CG dinucleotide on the same strand corresponds to a
consecutive $\{G, C\}$ pair in the transition sequence — a Lee-distance-2 bigram, i.e.
a complement bigram where $r_{i+1} = \theta(r_i)$.

**Prediction.** In a real natural-language corpus, complement bigrams (consecutive
symbols at Lee distance 2) are underrepresented relative to the null expectation.

**Null expectation.** For a uniformly random transition sequence, each successor is
drawn from the three symbols $\{A,B,C,D\} \setminus \{r_i\}$, exactly one of which is
$\theta(r_i)$. The null rate of complement bigrams is therefore $1/3 \approx 0.333$.

**Prediction.** Observed complement-bigram frequency in real embedding corpora is
$< 1/3$.

The intuition matches: a semantic trajectory that jumps to the full complement of its
current state in a single step represents an abrupt, maximum-Lee-distance transition.
Embedding distributions produced by trained models should prefer smaller steps,
suppressing complement bigrams relative to the uniform null.

**Falsification condition.** Observed complement-bigram frequency $\geq 1/3$ across
multiple corpora and models would falsify this prediction.

Test protocol: §T-1 (null baseline), §T-2 (code corpus).

---

## P4 — Transition/transversion asymmetry and weighted Lee distance

Under the CGAT mapping (P1), Lee-distance-1 pairs split into two chemically distinct
types:

- **Transitions** ($G$–$A$ and $C$–$T$): change the keto/amino bit only, remaining
  within the same ring class (purine-to-purine or pyrimidine-to-pyrimidine). In
  molecular evolution, transitions occur at rate $\mu_{\text{ts}}$.
- **Transversions type 1** ($G$–$T$ and $A$–$C$): change the purine/pyrimidine bit
  only, crossing ring classes while preserving functional group. Rate
  $\mu_{\text{tv1}} \approx \mu_{\text{ts}}/2$.
- **Complement transversions** ($G$–$C$ and $A$–$T$, Lee distance 2): both bits
  change. Rate $\mu_{\text{tv2}} \approx \mu_{\text{ts}}/4$.

The empirical transition-to-transversion ratio Ti/Tv is approximately 2–4 in
vertebrate genomes. The uniform Lee metric treats all Lee-distance-1 pairs
identically; the biological evidence implies that within Lee-distance-1, transitions
should be semantically cheaper than same-ring-class transversions.

**Prediction.** A weighted Lee metric with weights:

$$w_1 = 1 \text{ (transition)}, \quad w_2 \approx 1.5\text{–}2 \text{ (type-1 transversion)}, \quad w_3 \approx 3 \text{ (complement transversion)}$$

outperforms the uniform Lee metric ($w_1 = w_2 = 1$, $w_3 = 2$) on retrieval
benchmarks. The empirical weights should be estimable from the corpus's own
complement-bigram and transition-bigram frequencies, without reference to any external
gold standard.

**Falsification condition.** If the weighted metric performs no better than the
uniform metric on a held-out retrieval benchmark, the transition/transversion
distinction is not present in the embedding distributions produced by the models under
test.

Test protocol: §T-3 (embedding models).

---

## P5 — Reverse complement retrieval of semantic antonyms

In molecular biology, the *reverse complement* of a DNA sequence
$S = (s_0, s_1, \ldots, s_{n-1})$ is the sequence read on the antiparallel strand:

$$\bar{S}^R = (\theta(s_{n-1}),\ \theta(s_{n-2}),\ \ldots,\ \theta(s_0))$$

**Definition.** The *reverse complement* of a transition sequence
$R = (r_0, r_1, \ldots, r_{k-1})$ is:

$$\bar{R}^{\text{rev}} = (\theta(r_{k-1}),\ \theta(r_{k-2}),\ \ldots,\ \theta(r_0))$$

**Prediction.** Given a document $D$ with transition sequence $R$, the document whose
transition sequence is closest (under Lee distance) to $\bar{R}^{\text{rev}}$ is the
semantic antonym of $D$ — the document that traverses the same semantic axis in the
opposite direction.

The intuition follows from P1: the antiparallel strand in DNA encodes the complementary
information on the same axis. A document that argues $x \to \theta(x) \to x$ and a
document arguing $\theta(x) \to x \to \theta(x)$ traverse the same axis in opposite
orientations. Their transition sequences are reverse complements of each other.

**Concrete test.** Embed a set of antonym pairs (e.g. *optimism/pessimism*,
*ascent/descent*, *expansion/contraction*). For each document in the pair, compute the
reverse complement of its transition sequence, then query the index. The prediction is
that the query returns the antonym with higher frequency than an unrelated document at
the same Lee distance from the query.

This prediction is orthogonal to cosine similarity: two antonym documents can have
high cosine similarity (their embeddings are close on $S^{n-1}$) while being reverse
complements in transition-sequence space.

**Falsification condition.** Reverse-complement queries retrieve antonyms at no better
than chance rate relative to documents at the same Lee distance.

Test protocol: §T-3 (embedding models), §T-4 (local LLMs).

---

## P6 — Two-stage search architecture (PAM/seed analog)

CRISPR-Cas9 searches a genome in two stages:

1. **PAM recognition** — Cas9 scans for the 3-base PAM motif (e.g. NGG for
   *Streptococcus pyogenes* Cas9) using rapid diffusion. This is an O(1)-per-site
   lookup covering ~6 bits of specificity.
2. **Seed matching** — the 12 bases of the gRNA adjacent to the PAM (the *seed region*)
   must match the target with near-zero mismatch tolerance. The remaining 8 bases
   (*PAM-distal*) tolerate up to 5 mismatches.

The Q² 64-bit key has the same two-level structure by construction (§D-3.3):

- **High-order bits** ($r_0$–$r_5$, top 12 bits): encode the coarsest semantic
  distinctions, analogous to the PAM plus seed region.
- **Low-order bits** ($r_6$–$r_{31}$, bottom 52 bits): encode fine intra-block
  structure, analogous to the PAM-distal region.

**Prediction.** The optimal Q² search architecture is:

1. Hash lookup on the top $k$ bits of the key, $k \approx 12$–$24$, returning a
   small candidate set.
2. Full Lee-distance computation on the candidate set only.

The false-positive rate as a function of mismatch depth follows a sigmoid with
inflection near symbol position 12–14 from the MSB, not a linear function of Lee
distance. The optimal $k$ for Stage 1 is the point where the specificity of the
hash pre-filter exceeds the cost of the full Lee computation on surviving candidates.

**Mismatch tolerance asymmetry.** The high-order symbols (depth 0–5) should behave as
the seed region: retrieval recall degrades sharply with mismatches here. The low-order
symbols (depth 18–31) should behave as PAM-distal: mismatches are tolerated and
precision is maintained. The magnitude of the asymmetry is bounded by the Ti/Tv ratio
from P4.

**Falsification condition.** If a flat (uniform-depth) search performs equally well as
the two-stage architecture on latency-matched benchmarks, the seed/PAM-distal
asymmetry is absent from the key structure.

Test protocol: §T-3 (embedding models).

---

## P7 — Document secondary structure

RNA secondary structure is the set of non-crossing Watson–Crick base pairs that
minimises free energy. The Nussinov algorithm finds this in $O(n^3)$ by dynamic
programming: a pair $(i, k)$ is included in the structure if $s_i$ and $s_k$ are
complements, and no pair in the structure crosses $(i, k)$.

The same algorithm applies to transition sequences. Define a **complement pair** in a
transition sequence $R$ as a pair of positions $(i, k)$ with $i + 2 \leq k$ such that:

- $r_i = r_k$ (the flanking symbols are equal), and
- $r_{i+1} = \theta(r_i)$ and $r_{k-1} = \theta(r_k)$ (the interior opens and closes
  on the complement).

A set of non-crossing complement pairs is the *secondary structure* of $R$.

**Prediction.** Documents whose transition sequences admit a rich secondary structure
(many nested complement pairs) encode rhetorically complex argument forms — nested
qualifications, embedded counterarguments, recursive dialectical moves.

Two documents with identical 64-bit keys but different secondary structures have the
same coarse semantic address but different argumentative shape. Secondary structure
distinguishes a direct assertion from a dialectical one even when both resolve to the
same semantic neighbourhood.

**Pseudoknots as long-range cross-references.** A *pseudoknot* in RNA secondary
structure is a set of base pairs that cross — pairs $(i, k)$ and $(j, l)$ with
$i < j < k < l$. These cannot be represented as a planar tree; they require
$O(n^5)$–$O(n^6)$ algorithms to find optimally and are NP-hard in full generality. In
a transition sequence, a pseudoknot corresponds to a semantic cross-reference: a
document returns to an earlier topic after an intervening argument, crossing the
boundary of the intervening hairpin. Such structures should be rare and computationally
expensive to detect, consistent with the biological case.

**Falsification condition.** If secondary structure complexity (number of nested
complement pairs) is uncorrelated with human-annotated rhetorical complexity on the
same documents, the secondary structure prediction is falsified.

Test protocol: §T-3 (embedding models), §T-4 (local LLMs).

---

## P8 — Codon degeneracy and synonymous substitutions

The genetic code maps 64 codons to 20 amino acids. Most amino acids are encoded by 2–6
synonymous codons. Synonymous mutations change the codon without changing the encoded
amino acid. Their frequency is non-uniform and correlated with tRNA abundance (*codon
usage bias*).

In Q², a triplet of transition symbols $(r_i, r_{i+1}, r_{i+2})$ is a *semantic
codon*. Multiple distinct codons can encode the same *semantic primitive* — a
conceptual unit represented at the triplet level.

**Prediction.** The frequency distribution of transition triplets in a real corpus is
non-uniform and non-random: a small number of triplet forms account for a
disproportionate share of occurrences. The high-frequency triplets are the "synonymous
codons" for common semantic primitives (hedges, intensifiers, logical connectives,
topic-establishing patterns). Low-frequency triplets correspond to rare rhetorical
moves.

**Corpus-dependence.** The frequency distribution should vary by corpus domain in a
manner analogous to codon usage bias by organism: a code-specialised model should
exhibit different triplet frequency biases than a natural-language model, reflecting
different distributions of semantic primitives in the two domains.

**Synonymy density as a vocabulary measure.** Documents that use a wider variety of
triplet forms (lower synonymy concentration) are making more diverse semantic moves.
This gives a basis for a vocabulary-richness metric defined over transition sequences
rather than surface-form word counts.

**Falsification condition.** If the triplet frequency distribution is statistically
indistinguishable from a Zipf-distributed random sequence with the same bigram
constraints, the semantic-codon structure is not present.

Test protocol: §T-2 (code corpus), §T-3 (embedding models).

---

## P9 — Alphabet optimality (Kerdock/Preparata)

The Kerdock and Preparata codes are optimal non-linear binary error-correcting codes.
Under the bijection established by Hammons, Kumar, Calderbank, Sloane, and Solé
(1994), these codes become *linear* when lifted from $\{0,1\}^{2n}$ to $\mathbb{Z}_4^n$
via the Gray map. This is the theorem cited in §D-2.7.

The significance is that $\mathbb{Z}_4$ (2 bits per symbol) is the *minimum* ring over
which optimal non-linear binary codes become linear. Neither $\mathbb{Z}_2$ (1 bit)
nor $\mathbb{Z}_8$ (3 bits) has this property for the same code families.

**Prediction.** Increasing the alphabet to $\mathbb{Z}_8$ or $\mathbb{Z}_{16}$ (3 or
4 bits per symbol) provides diminishing retrieval returns, holding embedding dimension
constant. The algebraic advantage of the Gray-map isometry is specific to
$\mathbb{Z}_4$: for larger alphabets there is no exact Lee-to-Hamming isometry under
plain Hamming distance, and this complicates the `popcnt(XOR)` computation.

Specifically:

- $\mathbb{Z}_8$: the 3-bit Gray code cannot be an isometry from $(\mathbb{Z}_8, d_L)$
  to $(\{0,1\}^3, d_H)$ because $\max d_L = 4$ on $\mathbb{Z}_8$ while $\max d_H = 3$
  on 3 bits. The cyclic wrap $d_L(7,0) = 1$ maps to Gray codes $\texttt{100}$ and
  $\texttt{000}$ (Hamming distance 1), and adjacent pairs at the midpoint
  ($d_L(3,4)=1$, Gray codes $\texttt{010}$ and $\texttt{110}$, Hamming distance 1)
  also behave correctly, but larger Lee distances cannot be represented exactly by
  Hamming distance on 3 bits. Consequently, `popcnt(XOR)` over Gray-coded $\mathbb{Z}_8$
  symbols computes Hamming distance, which underestimates Lee distance for some pairs.
  To recover true Lee distance one must apply a correction step, e.g., decode each
  3-bit Gray symbol back to $\mathbb{Z}_8$ and use a small Lee-distance lookup table,
  or map bitwise differences through a per-symbol LUT/SIMD transform before summing.
- The information gain per symbol from $\mathbb{Z}_4$ to $\mathbb{Z}_8$ is
  $\log_2 8 - \log_2 4 = 1$ bit, but the additional bit encodes intra-cell position
  within the four magnitude classes, which §D-1.6 shows is not recoverable signal
  for L1 retrieval.

**Falsification condition.** If a $\mathbb{Z}_8$-encoded system with otherwise
identical architecture achieves a statistically significant improvement in retrieval
on a held-out benchmark, the $\mathbb{Z}_4$ optimality prediction is falsified. This
would imply that the fifth-through-eighth quantization cells carry retrievable semantic
signal, contradicting §D-1.6.

Test protocol: §T-3 (embedding models).

---

## P10 — Key entropy and collision rate

The 64-bit transition key is intended as a compact semantic address: most distinct
documents should map to distinct keys, and collisions should occur mainly between
semantically similar documents.

**Prediction.** For a corpus of size $N$ (e.g., 10k–100k documents), the empirical
collision rate of 64‑bit keys should be close to the random-hash expectation
$O(N^2 / 2^{65})$, and collision pairs should be more semantically similar (lower Lee
distance) than random pairs. A collision rate substantially higher than the null
expectation indicates loss of discriminative power in the key.

**Test protocol.** §T-2, §T-3. Compute key frequencies and estimate collision rate.
Compare against the theoretical baseline for uniform 64-bit keys, and inspect the
semantic similarity of collided pairs.

---

## Summary table

| ID | Prediction | From | Tested in | Effort |
|:--:|:-----------|:----:|:---------:|:------:|
| P1 | Gray code bits encode purine/pyrimidine × keto/amino axes | CGAT mapping | — (algebraic) | None |
| P2 | $\rho_{\text{hp}}$: Dialectical $>$ Direct $\approx 1/9 >$ Negated | RNA hairpin | §T-2, §T-3 | Low |
| P3 | Complement bigrams $< 1/3$ in real corpora | CpG suppression | §T-1, §T-2 | Low |
| P4 | Weighted Lee outperforms uniform Lee on retrieval | Ti/Tv ratio | §T-3 | Medium |
| P5 | Reverse-complement query retrieves semantic antonym | Antiparallel strand | §T-3, §T-4 | Low |
| P6 | Two-stage (hash + Lee) outperforms flat Lee search | PAM/seed architecture | §T-3 | Medium |
| P7 | Secondary structure complexity correlates with rhetorical complexity | RNA folding | §T-3, §T-4 | High |
| P8 | Triplet frequency distribution shows codon-usage-bias pattern | Genetic code degeneracy | §T-2, §T-3 | Low |
| P9 | $\mathbb{Z}_8$ encoding yields no significant retrieval improvement | Kerdock/Preparata | §T-3 | Medium |
| P10 | Key collision rate is low and correlates with semantic similarity | Key design | §T-2, §T-3 | Low |

P3 requires only a frequency count on quantizer output and can be run immediately once
the quantizer from [PR #9](https://github.com/devlux76/q2/pull/9) is merged. P2 and
P5 require a retrieval benchmark but no new infrastructure. P4, P6, and P9 require
benchmark-calibrated metric variants. P7 requires a Nussinov dynamic programming
implementation over transition sequences.
