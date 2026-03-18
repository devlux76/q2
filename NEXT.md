# NEXT

## Current status

- ✅ Added an OPFS-backed local file store (`src/opfs.ts`) that saves files by SHA-256 hash and keeps a filename/metadata mapping in `localStorage`.
- ✅ Added a small UI panel in the sidebar for:
  - Drag-and-drop file import
  - URL import
  - Listing stored files with Download + Delete actions
- ✅ Added tests for the OPFS store (`test/opfs.test.ts`) and updated app tests where needed.
- ✅ Refactored settings persistence into `src/settings.ts` and kept the existing public API by re-exporting helpers from `src/app.ts`.
- ✅ Added `src/q2stats.ts` — pure-TypeScript analysis functions for testing PREDICTIONS hypotheses:
  - `complement(z)` — θ involution: G↔C, A↔T (P1)
  - `leeDistance(a, b)` — Lee metric on Z₄ (P1)
  - `leeDistancePacked(a, b)` — popcount(XOR) over packed Q² bytes (P6)
  - `grayEncode / grayDecode` — Gray code round-trip (P1)
  - `unpackSymbols(packed, n)` — extract Z₄ symbols from packed bytes
  - `runReduce(symbols)` — run-length compress to transition sequence R
  - `hairpinDensity(seq)` — ρ_hp (P2); null expectation 1/9
  - `complementBigramFreq(seq)` — complement bigram frequency (P3); null expectation 1/3
  - `tripletFreqs(seq)` — triplet (codon) frequency distribution (P8)
  - `bigramFreqs(seq)` — bigram frequency distribution
  - `reverseComplementSeq(seq)` — reverse complement for antonym retrieval (P5)
  - `collisionStats(keys)` — 64-bit key collision rate (P10)
  - `nullCollisionExpectation(n)` — theoretical birthday-paradox baseline (P10)
  - `leeDistanceSeq(a, b)` — sum of per-symbol Lee distances
- ✅ Added `test/t0.test.ts` — T0 phase: unit tests and algebraic invariants (48 tests, no model download):
  - P1: complement involution properties (θ(θ(z))=z, fixed points, Watson–Crick pairs)
  - P1: Lee metric properties (symmetry, bounds, complement = max distance)
  - P1: Gray-code Lee-to-Hamming isometry (all 16 symbol pairs)
  - P1: Gray bits encode purine/pyrimidine and keto/amino axes
  - T0-Q: quantisation smoke tests (valid key for random vectors, determinism, round-trip)
  - T0-P10: key collision rate for 500 synthetic embeddings ≈ 0
  - Edge-case tests for unpackSymbols, runReduce, hairpinDensity, complementBigramFreq, tripletFreqs, reverseComplementSeq
- ✅ Added `test/t1.test.ts` — T1 phase: null-distribution baselines on synthetic data (13 tests):
  - P2: ρ_hp ≈ 1/9 for uniform random transition sequences (multiple lengths)
  - P3: complement bigram frequency ≈ 1/3 for uniform random sequences
  - P8: triplet entropy close to maximum, all 36 valid forms observed
  - Structural invariants of run-reduced transition sequences
- ✅ Fixed pre-existing TypeScript typecheck errors in `src/app.ts` and `src/opfs.ts`.
- ✅ All 113 tests pass (`bun run test`); TypeScript typecheck clean (`bun run typecheck`).

## What to do next

### 1) T2 — Structured code corpus (requires code embedding model, no LLM)
- Download or export a code-specialised embedding model to ONNX (e.g. a CodeBERT-class
  model) and add it to the worker/OPFS loader.
- Build a small TypeScript function corpus (~1 000 functions) from the project itself
  or open-source TypeScript repos and store it in OPFS.
- Run the retrieval benchmark (signature → implementation; call-site → callee body)
  and compute P2, P3, P8, P10 sub-tests using `src/q2stats.ts`.

### 2) T3 — Matryoshka / dedicated embedding models
- Add support for loading standard embedding models (e.g. `nomic-embed-text`) via
  `@huggingface/transformers` or ONNX, alongside the existing LFM model.
- Implement the probe corpus for P2 (30–100 Direct/Dialectical/Negated sentence
  triplets) and run the ρ_hp ordering test.
- Implement the antonym pair set for P5 (50–200 pairs from WordNet) and the
  reverse-complement antonym retrieval benchmark.

### 3) Weighted Lee distance (P4)
- Estimate Ti/Tv1/Tv2 bigram frequencies from a corpus using `bigramFreqs`.
- Add `weightedLeeDist(seq, weights)` to `src/q2stats.ts` and benchmark config D vs E.

### 4) Nussinov secondary structure (P7)
- Implement `nussinov(seq)` in `src/q2stats.ts` using dynamic programming.
- Cap at a 1 000-symbol window for tractability.
- Correlate secondary structure depth with rhetorical complexity annotations.

### 5) Make OPFS files usable for the pipeline
- Add a mechanism to load model/corpus artifacts from OPFS into the worker (e.g.
  pass a `localHash` to the worker and fetch from OPFS).
- Add UI to select a stored file as the model weights source.

## Notes for future context

- The project is a browser-only ONNX text-generation + Q² quantisation demo.
- Q² kernel is in `src/q2.wat` with a TS wrapper in `src/q2.ts`.
- Analysis functions for T0–T4 hypotheses are in `src/q2stats.ts`.
- Tests run via Vitest; browser tests run via Playwright.
- The existing architecture is `app.ts` (UI) + `worker.ts` (inference) + `q2.ts` (Q² logic) + `q2stats.ts` (analysis).
