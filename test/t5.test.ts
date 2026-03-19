/**
 * t5.test.ts — T5 Phase: Phylomemetic Fingerprinting (P14)
 *
 * Tests prediction P14 from TESTING.md §T5: author fingerprint stability,
 * RLHF entropy compression, cross-lineage influence detection, and temporal
 * ordering constraint. No model download required; sequences are generated
 * deterministically to exhibit the statistical properties predicted for real
 * author-attributed text embeddings.
 *
 * References: TESTING.md §T5, PREDICTIONS.md §P14 (issue #40).
 *
 * Test organisation:
 *   T5-P14a — Author fingerprint stability: Q² stats stable within author,
 *              separable across authors
 *   T5-P14b — RLHF entropy compression: AI model CV < human author CV
 *   T5-P14c — Cross-lineage influence: authors cluster by hairpin fingerprint
 *   T5-P14d — Temporal ordering: influenced author closer to earlier source
 */

import { describe, expect, it } from 'vitest';
import {
  complement,
  hairpinDensity,
  complementBigramFreq,
  leeDistanceSeq,
  tripletFreqs,
} from '../src/q2stats.ts';

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Seeded xorshift32 RNG. */
function makeRng(seed: number) {
  let s = (seed === 0 ? 123456789 : seed) >>> 0;
  return function () {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

/**
 * Simulate an "author-fingerprinted" sequence.
 *
 * Each synthetic author has a characteristic hairpin rate bias applied on top of a
 * base rate of 0.2. A high authorBias (e.g. 0.25) produces high hairpin density;
 * a negative bias (e.g. −0.15) suppresses hairpins — simulating stylistic diversity.
 */
function authorSeq(length: number, seed: number, authorBias: number): number[] {
  const rng = makeRng(seed);
  const hairpinRate = Math.max(0, Math.min(0.8, 0.2 + authorBias));
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    if (seq.length + 2 <= length && rng() < hairpinRate) {
      seq.push(complement(last));
      seq.push(last);
    } else {
      const others = [0, 1, 2, 3].filter((x) => x !== last);
      seq.push(others[Math.floor(rng() * others.length)]!);
    }
  }
  return seq.slice(0, length);
}

/**
 * Simulate RLHF-compressed sequences for T5/P14b.
 *
 * RLHF training compresses stylometric variance by averaging over many training
 * authors. The output resembles the population null (no systematic hairpin signal):
 * random uniform transitions from 3 non-same symbols, giving ρ_hp ≈ 1/9.
 * Per-document variation is small (sampling noise only), so CV is very low.
 */
function rlhfSeq(length: number, seed: number): number[] {
  // Uniform random transitions (no hairpin injection) → ρ_hp ≈ 1/9 by construction.
  const rng = makeRng(seed + 9000);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last);
    seq.push(others[Math.floor(rng() * others.length)]!);
  }
  return seq.slice(0, length);
}

/** Coefficient of variation (SD / mean). */
function cv(values: number[]): number {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;
  const variance = values.reduce((s, x) => s + (x - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance) / mean;
}

// ─── T5-P14a: Author fingerprint stability ─────────────────────────────────

describe('T5-P14a: author fingerprint stability — Q² stats stable within author, separable between', () => {
  const SEQ_LEN = 400;
  const N_DOCS = 40;

  it('within-author hairpin density has low variance (SD < 0.10)', () => {
    // Two authors with distinct biases (+0.2 and −0.1)
    const authorA = Array.from({ length: N_DOCS }, (_, seed) =>
      hairpinDensity(authorSeq(SEQ_LEN, seed, 0.2)),
    );
    const mean = authorA.reduce((a, b) => a + b, 0) / authorA.length;
    const sd = Math.sqrt(authorA.reduce((s, x) => s + (x - mean) ** 2, 0) / authorA.length);
    expect(sd).toBeLessThan(0.1);
  });

  it('cross-author hairpin density means are separable by >0.05', () => {
    const authorA = Array.from({ length: N_DOCS }, (_, seed) =>
      hairpinDensity(authorSeq(SEQ_LEN, seed, 0.2)),
    );
    const authorB = Array.from({ length: N_DOCS }, (_, seed) =>
      hairpinDensity(authorSeq(SEQ_LEN, seed + N_DOCS, -0.1)),
    );
    const meanA = authorA.reduce((a, b) => a + b, 0) / authorA.length;
    const meanB = authorB.reduce((a, b) => a + b, 0) / authorB.length;
    expect(Math.abs(meanA - meanB)).toBeGreaterThan(0.05);
  });

  it('author A (high bias) has mean ρ_hp > author B (low bias)', () => {
    const rhopsA: number[] = [];
    const rhopsB: number[] = [];
    for (let seed = 0; seed < N_DOCS; seed++) {
      rhopsA.push(hairpinDensity(authorSeq(SEQ_LEN, seed, 0.25)));
      rhopsB.push(hairpinDensity(authorSeq(SEQ_LEN, seed, -0.15)));
    }
    const meanA = rhopsA.reduce((a, b) => a + b, 0) / rhopsA.length;
    const meanB = rhopsB.reduce((a, b) => a + b, 0) / rhopsB.length;
    expect(meanA).toBeGreaterThan(meanB);
  });

  it('complement bigram frequency is stable within an author (low CV)', () => {
    const cbfs = Array.from({ length: N_DOCS }, (_, seed) =>
      complementBigramFreq(authorSeq(SEQ_LEN, seed, 0.2)),
    );
    const cvVal = cv(cbfs);
    // CV < 0.3 means the fingerprint is reasonably stable within an author
    expect(cvVal).toBeLessThan(0.3);
  });
});

// ─── T5-P14b: RLHF entropy compression ────────────────────────────────────

describe('T5-P14b: RLHF entropy compression — AI model CV < human author CV', () => {
  const SEQ_LEN = 400;
  const N_DOCS = 40;

  it('RLHF sequences have lower hairpin CV than human author sequences', () => {
    // Human author: bias varies by ±0.15 across documents (high variance)
    const humanRhops = Array.from({ length: N_DOCS }, (_, seed) =>
      hairpinDensity(authorSeq(SEQ_LEN, seed, (seed % 7 - 3) * 0.05)),
    );
    // RLHF model: tiny jitter (±0.02) — low variance
    const rlhfRhops = Array.from({ length: N_DOCS }, (_, seed) =>
      hairpinDensity(rlhfSeq(SEQ_LEN, seed)),
    );
    const humanCV = cv(humanRhops);
    const rlhfCV = cv(rlhfRhops);
    expect(rlhfCV).toBeLessThan(humanCV);
  });

  it('RLHF complement bigram CV is lower than human author CV', () => {
    const humanCbfs = Array.from({ length: N_DOCS }, (_, seed) =>
      complementBigramFreq(authorSeq(SEQ_LEN, seed, (seed % 7 - 3) * 0.05)),
    );
    const rlhfCbfs = Array.from({ length: N_DOCS }, (_, seed) =>
      complementBigramFreq(rlhfSeq(SEQ_LEN, seed)),
    );
    const humanCV = cv(humanCbfs);
    const rlhfCV = cv(rlhfCbfs);
    expect(rlhfCV).toBeLessThan(humanCV);
  });

  it('RLHF triplet-distribution entropy is lower-variance than human text', () => {
    // Entropy of the triplet distribution (bits per triplet type)
    function tripletEntropy(seq: number[]): number {
      const freqs = tripletFreqs(seq);
      const total = Object.values(freqs).reduce((a, b) => a + b, 0);
      if (total === 0) return 0;
      let h = 0;
      for (const count of Object.values(freqs)) {
        const p = count / total;
        if (p > 0) h -= p * Math.log2(p);
      }
      return h;
    }

    const humanEntropies = Array.from({ length: N_DOCS }, (_, seed) =>
      tripletEntropy(authorSeq(SEQ_LEN, seed, (seed % 7 - 3) * 0.05)),
    );
    const rlhfEntropies = Array.from({ length: N_DOCS }, (_, seed) =>
      tripletEntropy(rlhfSeq(SEQ_LEN, seed)),
    );
    const humanCV = cv(humanEntropies);
    const rlhfCV = cv(rlhfEntropies);
    // RLHF should have lower coefficient of variation in entropy
    expect(rlhfCV).toBeLessThan(humanCV);
  });

  it('RLHF mean ρ_hp is near null baseline (not systematically elevated)', () => {
    const rlhfRhops = Array.from({ length: N_DOCS }, (_, seed) =>
      hairpinDensity(rlhfSeq(SEQ_LEN, seed)),
    );
    const mean = rlhfRhops.reduce((a, b) => a + b, 0) / rlhfRhops.length;
    // RLHF model should produce near-null hairpin density (1/9 ± 0.08)
    expect(mean).toBeGreaterThan(1 / 9 - 0.08);
    expect(mean).toBeLessThan(1 / 9 + 0.08);
  });
});

// ─── T5-P14c: Cross-lineage influence detection ────────────────────────────

describe('T5-P14c: cross-lineage detection — authors cluster by Q² fingerprint', () => {
  const SEQ_LEN = 400;
  const DOCS_PER_AUTHOR = 10;

  it('mean within-author Lee distance < mean cross-author Lee distance', () => {
    const biases = [0.25, -0.05, 0.45];
    const authorCorpora = biases.map((bias, aIdx) =>
      Array.from({ length: DOCS_PER_AUTHOR }, (_, seed) =>
        authorSeq(SEQ_LEN, aIdx * 100 + seed, bias),
      ),
    );

    let withinSum = 0;
    let withinCount = 0;
    let crossSum = 0;
    let crossCount = 0;

    for (let a = 0; a < authorCorpora.length; a++) {
      // within-author pairs
      for (let i = 0; i < authorCorpora[a]!.length; i++) {
        for (let j = i + 1; j < authorCorpora[a]!.length; j++) {
          withinSum += leeDistanceSeq(authorCorpora[a]![i]!, authorCorpora[a]![j]!);
          withinCount++;
        }
      }
      // cross-author pairs
      for (let b = a + 1; b < authorCorpora.length; b++) {
        for (const seqA of authorCorpora[a]!) {
          for (const seqB of authorCorpora[b]!) {
            crossSum += leeDistanceSeq(seqA, seqB);
            crossCount++;
          }
        }
      }
    }

    const meanWithin = withinSum / withinCount;
    const meanCross = crossSum / crossCount;
    expect(meanCross).toBeGreaterThan(meanWithin);
  });

  it('author with high bias is further from author with low bias than from author with medium bias', () => {
    const highBias = 0.35;
    const medBias = 0.15;
    const lowBias = -0.1;

    const highSeqs = Array.from({ length: 8 }, (_, s) => authorSeq(SEQ_LEN, s, highBias));
    const medSeqs = Array.from({ length: 8 }, (_, s) => authorSeq(SEQ_LEN, s + 50, medBias));
    const lowSeqs = Array.from({ length: 8 }, (_, s) => authorSeq(SEQ_LEN, s + 100, lowBias));

    const distHighMed = highSeqs.reduce(
      (sum, h, i) => sum + leeDistanceSeq(h, medSeqs[i]!), 0,
    ) / highSeqs.length;
    const distHighLow = highSeqs.reduce(
      (sum, h, i) => sum + leeDistanceSeq(h, lowSeqs[i]!), 0,
    ) / highSeqs.length;

    expect(distHighLow).toBeGreaterThan(distHighMed);
  });

  it('k=3 nearest-author retrieval correctly identifies same author above chance', () => {
    // A query document should retrieve more same-author docs in top-3 than random chance (1/3)
    const biases = [0.3, -0.05, 0.5];
    const corpora = biases.map((bias, aIdx) =>
      Array.from({ length: 12 }, (_, seed) => authorSeq(SEQ_LEN, aIdx * 200 + seed, bias)),
    );
    let sameAuthorHits = 0;
    const queries = 10;
    for (let q = 0; q < queries; q++) {
      const authorIdx = q % 3;
      const query = authorSeq(SEQ_LEN, 9000 + q, biases[authorIdx]!);
      // Flatten corpus (excluding query author's last doc to avoid self-retrieval)
      const corpus: Array<{ seq: number[]; author: number }> = [];
      for (let a = 0; a < corpora.length; a++) {
        for (const s of corpora[a]!) {
          corpus.push({ seq: s, author: a });
        }
      }
      const ranked = corpus
        .map((c) => ({ author: c.author, dist: leeDistanceSeq(query, c.seq) }))
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 3);
      sameAuthorHits += ranked.filter((r) => r.author === authorIdx).length;
    }
    // At chance, we'd expect ~1 same-author doc in top-3 out of 3 authors
    const hitRate = sameAuthorHits / (queries * 3);
    expect(hitRate).toBeGreaterThan(1 / 3);
  });
});

// ─── T5-P14d: Temporal ordering constraint ────────────────────────────────

describe('T5-P14d: temporal ordering — influenced author closer to earlier source', () => {
  const SEQ_LEN = 400;
  const N_DOCS = 10;

  it('influenced author sequences are closer to early source than late source', () => {
    const earlyBias = 0.35;
    const lateBias = -0.05;
    // "Influenced" author: intermediate bias, but using earlyBias as dominant parent
    const inflBias = (earlyBias * 0.7 + lateBias * 0.3); // 70% influence from early

    const earlySeqs = Array.from({ length: N_DOCS }, (_, i) =>
      authorSeq(SEQ_LEN, i, earlyBias),
    );
    const lateSeqs = Array.from({ length: N_DOCS }, (_, i) =>
      authorSeq(SEQ_LEN, i + 100, lateBias),
    );
    const inflSeqs = Array.from({ length: N_DOCS }, (_, i) =>
      authorSeq(SEQ_LEN, i + 200, inflBias),
    );

    const distInflEarly = inflSeqs.reduce(
      (sum, s, i) => sum + leeDistanceSeq(s, earlySeqs[i]!), 0,
    ) / N_DOCS;
    const distInflLate = inflSeqs.reduce(
      (sum, s, i) => sum + leeDistanceSeq(s, lateSeqs[i]!), 0,
    ) / N_DOCS;

    // Influenced author should be closer to the early source (dominant parent)
    expect(distInflEarly).toBeLessThan(distInflLate);
  });

  it('mixture author with equal influences is equidistant from both parents (±20%)', () => {
    const earlyBias = 0.3;
    const lateBias = -0.1;
    const mixBias = (earlyBias + lateBias) / 2; // perfectly balanced

    const earlySeqs = Array.from({ length: N_DOCS }, (_, i) =>
      authorSeq(SEQ_LEN, i, earlyBias),
    );
    const lateSeqs = Array.from({ length: N_DOCS }, (_, i) =>
      authorSeq(SEQ_LEN, i + 100, lateBias),
    );
    const mixSeqs = Array.from({ length: N_DOCS }, (_, i) =>
      authorSeq(SEQ_LEN, i + 200, mixBias),
    );

    const distMixEarly = mixSeqs.reduce(
      (sum, s, i) => sum + leeDistanceSeq(s, earlySeqs[i]!), 0,
    ) / N_DOCS;
    const distMixLate = mixSeqs.reduce(
      (sum, s, i) => sum + leeDistanceSeq(s, lateSeqs[i]!), 0,
    ) / N_DOCS;

    // Both distances should be similar (within 20% relative difference)
    const maxDist = Math.max(distMixEarly, distMixLate);
    const relDiff = Math.abs(distMixEarly - distMixLate) / (maxDist || 1);
    expect(relDiff).toBeLessThan(0.2);
  });

  it('RLHF model is not closer to any single human author (compressed variance)', () => {
    // RLHF model should have roughly equal distance to all human author archetypes,
    // consistent with being trained across all of them and compressing variance.
    const biases = [0.35, -0.05, 0.5, -0.15, 0.15];
    const authorMeans: number[] = biases.map((bias, aIdx) => {
      const authorSeqs = Array.from({ length: N_DOCS }, (_, i) =>
        authorSeq(SEQ_LEN, aIdx * 100 + i, bias),
      );
      const rlhfDocSeqs = Array.from({ length: N_DOCS }, (_, i) =>
        rlhfSeq(SEQ_LEN, aIdx * 100 + i),
      );
      return rlhfDocSeqs.reduce(
        (sum, r, i) => sum + leeDistanceSeq(r, authorSeqs[i]!), 0,
      ) / N_DOCS;
    });
    const minDist = Math.min(...authorMeans);
    const maxDist = Math.max(...authorMeans);
    // RLHF distances to authors should be within a 30% relative range (equidistant)
    const spread = (maxDist - minDist) / (maxDist || 1);
    expect(spread).toBeLessThan(0.3);
  });
});
