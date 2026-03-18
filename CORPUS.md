# Q² Testing: Model & Corpus Recommendations

**Scope:** Sub-1B ONNX models mapped to test phases T1–T4, plus a corpus layout targeting predictions P2–P13, with a total raw-text budget under 1 GB.

-----

## 1. Model Recommendations by Test Phase

All models listed below are under 1B parameters and available in ONNX format via Hugging Face (Transformers.js or the HF Hub directly).

### T1 / T0 — Null and Sanity Checks

These phases are primarily about establishing null distributions on random text, so the “real” embedding models from later phases are reused here. No special architecture is required.

**Standard text embedding (tiny, fast)**
`Xenova/all-MiniLM-L6-v2` (or `onnx-models/all-MiniLM-L6-v2-onnx`)

This is a BERT-mini encoder producing 384-dim embeddings, published specifically for Transformers.js / ONNX Runtime. Its role here is as a “boring” baseline to confirm the Q² pipeline introduces no embedding artifacts before real corpora are applied. It is used to establish null distributions for complement bigrams, triplets, and key entropy (P3, P8, P10) and to verify that retrieval on random text performs at chance.

-----

### T2 — Structured Code Corpus (TypeScript / JS)

The goal is TypeScript-oriented code embeddings with strong semantic signal and ONNX support.

**Primary code embedding model**
`sailesh27/unixcoder-base-onnx`

An ONNX export of Microsoft UniXcoder base, exposed for Transformers.js as a feature-extraction pipeline. It is a RoBERTa-style encoder producing 768-dim code/text embeddings, designed for semantic code search and documented to achieve 20–30% better code search accuracy than generic embeddings. Well under 1B parameters (BERT-base class).

**Alternative / comparison models**

- `onnx-community/codebert-base-ONNX` — ONNX version of `microsoft/codebert-base`, trained on CodeSearchNet for code+NL. Good classic CodeBERT-class baseline for code retrieval and P8 triplet distributions.
- `onnx-community/codebert-javascript-ONNX` — ONNX version of `neulab/codebert-javascript`, trained on JavaScript from `codeparrot/github-code-clean`. Useful for a heavily JS-oriented TypeScript corpus.

**Code-centric generative model (also relevant to T4)**
`onnx-community/Qwen2.5-Coder-0.5B-Instruct`

At 0.5B parameters, this is the generative coding baseline for T4 and provides an independent semantic sanity check for T2 (“are these code embeddings grouping correctly?”). A working ONNX version is explicitly available for Transformers.js.

**Recommended T2 configuration:** Use `unixcoder-base-onnx` as primary and `codebert-base-ONNX` as secondary for retrieval; use `Qwen2.5-Coder-0.5B-Instruct` for generative sanity checks. Together they cover the core retrieval benchmark (function-signature → body, call-site → callee) and predictions P2, P3, P8, and P10.

-----

### T3 — Matryoshka and Dedicated Embedding Models

Three model tiers are recommended to cover standard, Matryoshka, and tiny baselines.

**Standard strong embedding (non-Matryoshka)**
`mixedbread-ai/mxbai-embed-large-v1`

A BERT-large-class encoder, SOTA for its size on MTEB, competitive with models several times larger. An `onnx/` subfolder provides quantized weights (`model_quantized.onnx`), and it is fully usable through Transformers.js feature-extraction.

**Matryoshka embedding models**

1. `nomic-ai/nomic-embed-text-v1.5` — Long-context (8192-token) text encoder with explicit Matryoshka Representation Learning. Recommended truncation sizes: 768, 512, 256, 128, 64. The repo includes ONNX exports and a quantized ONNX build; the dedicated fork `michael-sigamani/nomic-embed-text-onnx` targets this family explicitly. Approximately 110M parameters — well within the 100–800M target range. This is the natural anchor for T3 Matryoshka tests.
1. `onnx-community/embeddinggemma-300m-ONNX` — A 300M-parameter embedding model derived from Gemma 3, producing 768-dim embeddings with Matryoshka sizes 512/256/128 supported. Distributed directly as ONNX (`onnx/model.onnx` + `model.onnx_data`) with first-class Transformers.js and ONNX Runtime examples.
1. `onnx-models/all-MiniLM-L6-v2-onnx` (tiny baseline) — 384-dim MiniLM, approximately 0.08 GB ONNX. Answers the question: “how much does Q² still help when embeddings are very small?”

**Recommended T3 configuration (configs A–E from TESTING.md):**

- **A:** Cosine on `all-MiniLM-L6-v2` (fast baseline)
- **B:** Q² + uniform Lee on `all-MiniLM-L6-v2`
- **C:** Cosine on `nomic-embed-text-v1.5`
- **D:** Q² + uniform Lee on `nomic-embed-text-v1.5`
- **E:** Q² + weighted Lee (P4) on `nomic-embed-text-v1.5`

Then repeat D/E with `embeddinggemma-300m-ONNX` to confirm that P2/P3/P4/P5/P7 are not Nomic-specific behaviors.

-----

### T4 — Standard Local LLMs (< 1B)

These are decoder-only models from which final hidden states are extracted and mean-pooled. They are used to check whether P2/P3/P5/P7 survive when activations come from a general or code LLM rather than a retrieval-tuned encoder — that is, whether hairpin and antonym reverse-complement effects are universal transformer phenomena or specific to retrieval training.

**General language model**
`onnx-community/Qwen2-0.5B-Instruct-ONNX`

At 0.5B parameters, this is a capable general instruction model. The documentation explicitly describes ONNX runtime and Transformers.js usage for text generation.

**Code-centric LLM**
`onnx-community/Qwen2.5-Coder-0.5B-Instruct` (as above)

Provides a second activation geometry tuned heavily on code.

**Recommended T4 configuration (configs F/G from TESTING.md):**

- **F:** Cosine on pooled `Qwen2-0.5B` states
- **G:** Q² + uniform Lee on the same activations; optionally run a parallel track with `Qwen2.5-Coder-0.5B`

> **Note on the 1B ceiling:** If the constraint is relaxed, `LiquidAI/LFM2.5-1.2B-Instruct-ONNX` is an available option, but it is recommended to keep it out of the first test matrix.

-----

## 2. Corpus Layout

The complete corpus stays comfortably under 1 GB of raw text, leaving headroom for indices and probe metadata. Sizes below are design targets enforced by random sampling.

### C0 — Synthetic Random Text (~0 MB)

Generated at test time from PRNG seeds; nothing stored on disk.

**Purpose:** T1 null distribution. Calibrates complement bigrams, triplets, and key entropy for each embedding model.

**Predictions exercised:** P2, P3, P8, P10.

**How it proves/falsifies:** Confirms null baseline and complement bigram rate ≈ 1/3 for random transitions. Detects any encoding artifacts that already break P8 triplet uniformity before real corpora are introduced.

-----

### C1 — General Natural Language (~200 MB)

**Source:** `togethercomputer/RedPajama-Data-1T`, sampling from books, Wikipedia, StackExchange, and C4 slices.

**Phases:** T1 (sanity vs. null), T3, T4.

**Predictions exercised:** P2, P3, P8, P10, P11, P12, P13 (monolingual side).

**How it proves/falsifies:**

- **P3:** Measures complement bigram frequency vs. the C0 null in broad real-world text. Underrepresentation (< 1/3) across slices supports complement suppression.
- **P2, P8:** Computes hairpin density and triplet distributions on longform documents vs. null; dialectical passages in StackExchange and Wikipedia discussions should show elevated rates.
- **P10:** Key collision rate on a 10–50k document sample.
- **P11–P13:** Provides the English/formal-writing baseline to contrast with spoken transcripts (C7) and cross-lingual content (C6).

-----

### C2 — Structured Code Corpus (~100 MB)

**Source:** `codeparrot/github-code-clean`, filtered to JavaScript and TypeScript files, extracting one short standalone function per sample. Target: 1,000–10,000 functions as specified in TESTING.md.

**Phases:** T2; some statistics reused in T3.

**Predictions exercised:** P2, P3, P8, P10, plus the T2 core retrieval benchmark.

**How it proves/falsifies:**

- **Core T2 retrieval:** Function-signature and call-site queries with ground truth from the AST validate the full Q² pipeline with code-specialized embeddings.
- **P2:** Functions are split into “call-and-return” vs. strictly linear via AST; elevated hairpin density in call-and-return functions relative to linear ones is directly falsifiable from the AST.
- **P3:** Complement bigram suppression in a domain where semantics are highly structured but phonology is irrelevant; if suppression still appears, that is strong evidence the effect is about embedding geometry, not text surface.
- **P8:** Codon-usage-like triplet frequency patterns tied to common code motifs (loops, branches).
- **P10:** Key entropy and collision rates in a machine-verifiable semantic domain.

-----

### C3 — Retrieval Benchmark Subset / BEIR (~250–300 MB)

**Source:** `BeIR/beir` on Hugging Face. Recommended tasks: MSMARCO passage, Natural Questions, FiQA, and SciFact. Each task is subsampled to approximately 75 MB of raw text, yielding 200–300 MB total across 3–4 tasks.

**Phases:** T3, T4 (core retrieval).

**Predictions exercised:** P2–P7, P9, P10.

**How it proves/falsifies:**

- **Core T3/T4 metrics:** nDCG@10 and Recall@100 for configs A–E and F–G, as assumed in TESTING.md.
- **P4:** Uniform vs. weighted Lee comparison at equal compute; weights inferred from bigram statistics on C1 and C3.
- **P5:** A small curated antonym corpus can be overlaid on BEIR documents rather than maintained as a separate index.
- **P6:** Two-stage vs. flat Lee search on a manageable but realistically noisy retrieval workload.
- **P7:** BEIR contains a mix of question types and document styles, supporting correlation of secondary-structure scores with rhetorical complexity where annotations exist (e.g., SciFact rationales).

-----

### C4 — Rhetorical Complexity / Argumentation Essays (~50 MB)

**Source:** Feedback Prize – Persuasive Essays (PERSUADE corpus), the basis of several rhetorical-structure papers. Target: a few thousand essays.

**Phases:** T3, T4.

**Predictions exercised:** P2, P7 (primary), P13 (partial).

**How it proves/falsifies:**

- **P7:** Existing argumentative element and discourse annotations serve as a proxy for rhetorical complexity. Nussinov-style secondary structure is computed over Q² transition sequences and correlated with those complexity labels.
- **P2:** Dialectical, direct, and negated argumentative moves frequently co-occur in the same essay sets, enabling probe triplets for the hairpin ordering test.
- **P13:** A small number of calibration triplets (e.g., sea/see/si-style) can be embedded here as short poems or essays.

-----

### C5 — Antonym Documents Corpus (≤10 MB)

**Source:** Curated. Build 50–200 antonym pairs (e.g., optimism/pessimism, ascent/descent, expansion/contraction) using WordNet or similar, each paired with a 1–3 paragraph explanatory text (Wikipedia-style or small artisan corpus).

**Phases:** T3, T4.

**Predictions exercised:** P2 (dialectical vs. negated forms), P5 (reverse-complement retrieval), P13 (negative regime scores for antonym pairs).

**How it proves/falsifies:**

- **P5:** Directly measures how often reverse-complement queries retrieve the antonym document compared to unrelated documents at the same Lee distance.
- **P2:** Direct, dialectical, and negated formulations are designed per antonym concept to stress-test the hairpin density ordering.

-----

### C6 — Cross-Lingual Matched-Content Corpus (~75–100 MB)

**Source:** `GEM/wiki_lingua` for multi-language aligned how-to articles, or aligned Wikipedia article sets across languages using Wikimedia/HF Wikipedia dumps. Target: balanced across 3–5 language pairs.

**Phases:** T3, T4 (multilingual extension).

**Predictions exercised:** P11, P12, P13.

**How it proves/falsifies:**

- **P11:** Compares complement suppression rates and bigram distributions across typologically different languages, controlling for content via aligned articles.
- **P12:** Cross-lingual matched articles describing the same physical or structural reality are used as conceptual near-neighbor pairs; checks whether Lee distance is lower than cosine distance suggests.
- **P13:** Computes regime scores for cross-lingual pairs vs. monolingual ones, expecting a right-skewed distribution for the former.

-----

### C7 — Spoken vs. Written Language Corpus (~50 MB)

**Source:** Any open ASR transcript corpus hosted on Hugging Face (TED Talks, conversational speech, etc.), contrasted against C1 and C3 written text. The exact spoken dataset can be determined later; the key distinction is “speech transcript” vs. “formal writing.”

**Phases:** T3, T4.

**Predictions exercised:** P11, P12.

**How it proves/falsifies:**

- **P11:** Compares complement-bigram suppression strength in spoken transcripts vs. formal writing (RedPajama books, Wikipedia, and legal-style slices). Stronger suppression in spoken text supports the articulatory grounding story.
- **P12:** Checks whether suppression rates cluster by phonotactic strictness vs. conceptual content across languages.

-----

## 3. Size Budget Summary

|Segment  |Description                                 |Target Size    |
|---------|--------------------------------------------|---------------|
|C0       |Synthetic random text                       |~0 MB          |
|C1       |General natural language (RedPajama)        |~200 MB        |
|C2       |Structured code corpus (JS / TS)            |~100 MB        |
|C3       |Retrieval benchmark subset (BEIR)           |~250–300 MB    |
|C4       |Rhetorical complexity / argumentation essays|~50 MB         |
|C5       |Antonym documents corpus                    |≤10 MB         |
|C6       |Cross-lingual matched-content corpus        |~75–100 MB     |
|C7       |Spoken vs. written language corpus          |~50 MB         |
|**Total**|                                            |**~735–810 MB**|

This leaves comfortable headroom for indices and probe corpora within the 1 GB budget.

-----

## 4. Prediction-to-Model-and-Corpus Matrix

|Prediction                                |Primary Corpora                        |Primary Models                                                     |
|------------------------------------------|---------------------------------------|-------------------------------------------------------------------|
|P2 (hairpin density)                      |C0 (null), C2 (code), C1/C3/C4 (probes)|All — MiniLM (null), UniXcoder (T2), Nomic / EmbeddingGemma (T3/T4)|
|P3 (complement suppression)               |C0 (null), C1, C2, C3, C7              |All embedding models; spoken vs. written comparison via C7         |
|P4 (weighted Lee)                         |C1 + C3                                |Nomic, EmbeddingGemma                                              |
|P5 (reverse-complement antonyms)          |C5 (primary), C3 (secondary)           |Nomic, EmbeddingGemma (T3); Qwen2 activations (T4)                 |
|P6 (two-stage hash + Lee search)          |C3                                     |Nomic, EmbeddingGemma with Q² keys                                 |
|P7 (secondary structure)                  |C4 (primary), C1 (longform)            |Nomic, EmbeddingGemma                                              |
|P8 (codon usage bias)                     |C0, C1, C2                             |MiniLM, Nomic, EmbeddingGemma, UniXcoder                           |
|P9 (Z₄ vs. Z₈)                            |C1 + C3                                |Nomic or EmbeddingGemma with Q² vs. Z₈ variant                     |
|P10 (key entropy / collisions)            |C2 (code), C1/C3 (NL)                  |All models                                                         |
|P11–P13 (regime, grounding, cross-lingual)|C1 vs. C7, C6                          |Nomic, EmbeddingGemma, MiniLM, Qwen2 activations                   |

-----

## 5. Suggested Next Step

A concrete model matrix with rows corresponding to `{MiniLM, Nomic, EmbeddingGemma, mxbai, UniXcoder, Qwen2, Qwen2.5-Coder}` and columns corresponding to `{T1–T4, P2–P13}` would clarify exactly which model–phase combinations need to be run vs. which can be pruned for an MVP.