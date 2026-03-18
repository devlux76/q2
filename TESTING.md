# Quaternary Semantic Quantization: Testing

> **Related documents:** [DESIGN.md](DESIGN.md) · [PREDICTIONS.md](PREDICTIONS.md)

Section references of the form §D-x.y refer to [DESIGN.md](DESIGN.md).
Section references of the form §P-x refer to [PREDICTIONS.md](PREDICTIONS.md).

---

## Contents

- [Overview](#overview)
- [T1 — Random text null distribution](#t1--random-text-null-distribution)
- [T2 — Structured code corpus](#t2--structured-code-corpus)
- [T3 — Matryoshka and dedicated embedding models](#t3--matryoshka-and-dedicated-embedding-models)
- [T4 — Standard local LLMs](#t4--standard-local-llms)
- [Cross-phase prediction matrix](#cross-phase-prediction-matrix)

---

## Overview

The four phases progress from the least constrained (random text, no semantic
structure) to the most constrained (code, where ground truth is machine-verifiable)
and then broaden to measure what the encoding adds over current best-practice
embedding models.

The expected performance ordering across phases is:

$$\text{T1 (random)} \ll \text{T4 (LLM, no enc)}
< \text{T4 (LLM + enc)} \lesssim \text{T3 (matryoshka, no enc)}
\leq \text{T3 (matryoshka + enc)} \ll \text{T2 (code + enc)}$$

Each phase carries a **core retrieval benchmark** and one or more **prediction
sub-tests** tied to specific claims in [PREDICTIONS.md](PREDICTIONS.md). The
prediction sub-tests are independent of the retrieval benchmark: they can confirm
or falsify a prediction even when retrieval accuracy is at chance.

## T0 — Unit tests and invariants

These are lightweight tests that should pass locally without large corpora or
model downloads. They validate the core Q² pipeline invariants and the algebraic
properties underpinning the predictions.

- **CGAT mapping / complement involution (P1).** Verify the Gray encoding and
  complement operation ($\theta$) satisfy the required symmetry: the complement of
  each symbol is the symbol with all Gray bits flipped, and the mapping exposes the
  two chemical axes (purine/pyrimidine and keto/amino).
- **Quantisation and packing invariants.** The existing unit tests in `test/q2.test.ts`
  cover these, but add a smoke test that random normal vectors always produce a valid
  64-bit key and that the key is stable for the same input.
- **Key entropy and collision rate (P10).** A small synthetic corpus (e.g. 10k random
  embeddings) should produce a key collision rate close to the uniform baseline.

These tests are quick and form the foundation for the higher-level phases.

---

## T1 — Random text null distribution

**Purpose.** Establish the distribution of transition sequence statistics that arise
from text with no semantic structure. Deviations from the null in later phases are
only interpretable if the null is measured, not assumed.

**Corpus generation.** Sample characters i.i.d. from the empirical English
character-frequency distribution (the 26-letter alphabet plus space and common
punctuation, weighted by standard corpus frequencies). Generate documents of varying
lengths (50, 200, 500, 2 000 characters). No LLM is involved in generation; the embedding
model serves only as a deterministic function of its input.

**Pipeline.** Apply the full Q² pipeline to each document:
1. Embed with the target model.
2. Quantize using empirically calibrated $\tau^*$ (§D-2.5).
3. Run-reduce to the transition sequence $R$ (§D-3.1).
4. Compute $K$ (§D-3.2), $\rho_{\text{hp}}$ (§P-2), bigram frequencies (§P-3),
   and triplet frequencies (§P-8).

**Core benchmark.** Retrieval precision on random-text corpora should be at chance
($\approx 1/C$ for a corpus of $C$ documents). Deviations indicate bias in the
encoding itself and must be diagnosed before proceeding.

**Prediction sub-tests.**

| Prediction | Measurement | Expected result |
|:----------:|:------------|:----------------|
| §P-3 | Complement-bigram frequency | $< 1/3$ if embedding model suppresses complement transitions; $\approx 1/3$ if not |
| §P-2 | $\rho_{\text{hp}}$ across document lengths | $\approx 1/9$ for all lengths (null baseline confirmation) |
| §P-8 | Triplet frequency distribution | Approximately uniform; any deviation flags an encoding artifact |

**Calibration output.** This phase produces the null distributions for $\rho_{\text{hp}}$,
complement-bigram frequency, and triplet entropy. All later-phase results are
interpreted relative to these distributions.

---

## T2 — Structured code corpus

**Purpose.** Validate the pipeline on a corpus with machine-verifiable ground truth,
using a code-specialised embedding model. Retrieval accuracy should be near-perfect;
failure here indicates a pipeline defect, not a semantic limitation.

**Corpus.** Short, well-typed functions from a single programming language (TypeScript
recommended, consistent with the existing codebase). Each function is independently
embeddable. Corpus size: 1 000–10 000 functions.

**Model.** A code-specialised embedding model exported to ONNX (e.g. a
CodeBERT-class or StarCoder-class model). The model choice fixes the activation
distribution; the threshold $\tau^*$ is calibrated from this corpus.

**Retrieval tasks.**
- Given a function signature, retrieve the implementation.
- Given a call site (a caller and the called function's name), retrieve the called
  function's body.

Ground truth is available from the AST without annotation.

**Core benchmark.** Retrieval accuracy at $k=1$ and $k=5$ (top-1 and top-5 recall).
Target: $> 90\%$ top-5 recall. Failure at this level indicates a defect in the Q²
pipeline.

**Prediction sub-tests.**

| Prediction | Measurement | Expected result |
|:----------:|:------------|:----------------|
| §P-2 | $\rho_{\text{hp}}$ for call-and-return functions vs. linear functions | Elevated $\rho_{\text{hp}}$ for functions that call helpers and return, relative to linear (no-call) functions of similar length |
| §P-3 | Complement-bigram frequency | $< 1/3$ if the code embedding model suppresses complement transitions |
| §P-8 | Triplet frequency distribution | Non-uniform; high-frequency triplets correspond to common code patterns (loop, branch, return) |
| §P-10 | 64-bit key collision rate | Collision rate close to the random-hash baseline; collisions should preferentially occur between semantically similar functions |

**AST ground truth for §P-2.** A function that calls one or more helpers and returns
to the caller has a call-and-return structure verifiable from the AST. No subjective
annotation is required. The prediction (elevated $\rho_{\text{hp}}$) is therefore
mechanically falsifiable.

---

## T3 — Matryoshka and dedicated embedding models

**Purpose.** Measure the contribution of quaternary encoding relative to
state-of-the-art retrieval-optimised models, including matryoshka (multi-scale
nested) embeddings. This phase also provides the strongest test of the hairpin
and antonym predictions, because the models are optimised for semantic structure.

**Models.** Two model types:
- Standard embedding model (e.g. `nomic-embed-text`, `all-MiniLM-L6-v2`)
- Matryoshka embedding model (e.g. `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`)

**Configurations evaluated.**

| Config | Encoding | Model |
|:------:|:--------:|:-----:|
| A | None — float cosine | Standard |
| B | Q² + uniform Lee | Standard |
| C | None — float cosine | Matryoshka |
| D | Q² + uniform Lee | Matryoshka |
| E | Q² + weighted Lee (§P-4) | Matryoshka |

**Core benchmark.** A standard retrieval benchmark (e.g. BEIR: MSMARCO, TREC-COVID,
NQ). Metrics: nDCG@10, Recall@100. Expected ordering: $E \geq D > C \geq B > A$.

**Prediction sub-tests.**

| Prediction | Measurement | Configurations | Expected result |
|:----------:|:------------|:--------------:|:----------------|
| §P-2 | $\rho_{\text{hp}}$ on probe corpus (Direct / Dialectical / Negated) | B, D | Dialectical $>$ Direct $\approx 1/9 >$ Negated |
| §P-3 | Complement-bigram frequency | B, D | $< 1/3$ |
| §P-4 | E vs. D retrieval | D, E | E $>$ D with statistical significance |
| §P-5 | Reverse-complement antonym retrieval on antonym pair set | B, D | Antonym retrieved at above-chance rate |
| §P-6 | Two-stage (hash + Lee) vs. flat Lee search, latency-matched | B, D | Two-stage $\geq$ flat on precision at equal latency |
| §P-7 | Secondary structure complexity vs. human rhetorical complexity annotation | B, D | Positive correlation; nested-argument documents score higher |
| §P-9 | $\mathbb{Z}_8$ vs. $\mathbb{Z}_4$ retrieval | B (Z₄), B′ (Z₈) | No significant improvement for Z₈ |
| §P-10 | 64-bit key collision rate | B, D | Collision rate close to random-hash baseline; collisions correlate with semantic similarity |

To keep the P6 measurements tractable, limit the two-stage search experiments to a
fixed prefix length (e.g. top 12–16 bits) and a manageable subset of queries, and
compare the precision/recall of the two-stage pipeline against a brute-force Lee
scan on the same candidate set.

For §P-4 (weighted Lee), estimate weights from observed bigram frequencies in the
corpus: compute empirical frequencies for transition (Ti) and transversion (Tv1/Tv2)
bigram types, then set weights proportional to the inverse frequency or log-odds
(relative to the uniform null). Evaluate a small grid around the estimated weights.

**Probe corpus for §P-2.** Construct 30–100 sentence triplets, one from each class
(Direct, Dialectical, Negated), on diverse topics. Triplets are matched by topic so
that the only varying factor is rhetorical stance. Measure $\rho_{\text{hp}}$ for each
document and test the ordering using a Wilcoxon signed-rank test (paired by topic).

**Antonym pair set for §P-5.** Curate 50–200 antonym pairs from a standard lexical
resource (WordNet). For each document in the pair, compute the reverse-complement
transition sequence, query the index, and record whether the antonym is retrieved
within the top $k$ results. Compare to a matched-difficulty non-antonym baseline.

**Annotation corpus for §P-7.** Select 50–100 documents with existing rhetorical
complexity annotations (e.g. from argumentation-mining datasets such as Persuasive
Essays (PE) or IBM Claim Stance). Compute the secondary structure of each document's
transition sequence using the Nussinov algorithm (§P-7), and measure Spearman rank
correlation between the number of nested complement pairs and the human-annotated
rhetorical complexity score.

> **Tractability note:** Nussinov is $O(n^3)$, so cap analysis at a fixed window size
> (e.g. the first 1k–2k transition symbols) or analyse a random 1k-symbol subsequence
> per document. For longer texts, a greedy heuristic (pairing the first valid complement
> in a sliding window) can provide an approximate secondary-structure score.

---

## T4 — Standard local LLMs

**Purpose.** Establish the floor contribution of Q² encoding when activations come
from a general-purpose LLM rather than a retrieval-optimised model.

**Models.** General-purpose local LLMs with accessible intermediate activations:
- LFM-2.5 1.2B (Liquid AI)
- Qwen2.5-1.5B (Alibaba)
- Any other model exportable to ONNX or accessible via `transformers`

**Activation source.** Final hidden-layer activations, mean-pooled over token
positions, L2-normalised. No retrieval fine-tuning.

**Configurations evaluated.**

| Config | Encoding | Model |
|:------:|:--------:|:-----:|
| F | None — float cosine | LLM |
| G | Q² + uniform Lee | LLM |

**Core benchmark.** Same benchmark as T3. Expected: $G > F$, but both below T3 configs
A–D (retrieval-optimised models are better retrieval models). The prediction is
that encoding raises performance from near-random toward the unencoded dedicated model
baseline.

**Prediction sub-tests.**

| Prediction | Measurement | Expected result |
|:----------:|:------------|:----------------|
| §P-2 | $\rho_{\text{hp}}$ on probe corpus | Signal present but noisier than T3; statistically significant enrichment in Dialectical class vs. null baseline ($1/9$) |
| §P-3 | Complement-bigram frequency | $< 1/3$; suppression may be weaker than in T3 |
| §P-5 | Reverse-complement antonym retrieval | Above-chance retrieval; effect size smaller than T3 |
| §P-7 | Secondary structure complexity vs. rhetorical complexity annotation | Positive correlation present but weaker than T3; statistical significance depends on model quality |

**If §P-2 sub-test fails in T4 but passes in T3:** the hairpin signal is present in
retrieval-optimised activation spaces but not in general-purpose LLM activations. This
is informative about what retrieval fine-tuning does to the structure of the L1 ball.

**If §P-2 sub-test passes in T4:** the hairpin signal is a property of the Q² encoding
applied to any sufficiently trained transformer, not an artefact of retrieval
optimisation.

---

## Cross-phase prediction matrix

| Prediction | T1 | T2 | T3 | T4 |
|:----------:|:--:|:--:|:--:|:--:|
| §P-2 Hairpin density ordering | Null baseline | Hairpin in call-and-return code | Full probe corpus test | Noisy probe test |
| §P-3 CpG suppression | Null baseline | Confirm or falsify | Confirm at scale | Partial test |
| §P-4 Weighted Lee | — | — | Full benchmark | — |
| §P-5 Reverse complement = antonym | — | — | Full antonym test | Partial test |
| §P-6 Two-stage search | — | — | Latency-matched benchmark | — |
| §P-7 Secondary structure | — | — | Correlation with annotation | Correlation (noisier) |
| §P-8 Codon usage bias | Null distribution | Domain-specific biases | Multi-domain comparison | — |
| §P-9 Z₈ optimality | — | — | Z₄ vs. Z₈ benchmark | — |
| §P-10 Key collision rate | — | Collision-rate benchmark | Collision-rate benchmark | — |

Tests in the T1 column establish baseline distributions. A prediction is
**confirmed** when the T3 or T4 result is statistically significant relative to the
T1 null. A prediction is **falsified** when the result is not distinguishable from
the null at the 95% confidence level across at least two model/corpus combinations.
