/**
 * t4.test.ts — T4 Phase: Standard Local LLMs
 *
 * Tests the prediction sub-tests from TESTING.md §T4 using synthetic transition
 * sequences designed to simulate the noisier statistical properties expected from
 * general-purpose LLM activations (as opposed to retrieval-optimised models in T3).
 * No model download is required.
 *
 * The key distinction from T3: predictions should still hold, but with reduced
 * effect sizes (wider tolerance bands, more variance across samples).
 *
 * References: TESTING.md §T4, PREDICTIONS.md §P2, §P3, §P5, §P7.
 *
 * Test organisation:
 *   T4-P2  — ρ_hp signal present in LLM activation sequences, but noisier
 *   T4-P3  — Complement-bigram suppression still present (< 1/3) in LLM sequences
 *   T4-P5  — Reverse-complement antonym retrieval: above-chance, effect smaller
 *   T4-P7  — Secondary structure complexity: positive correlation, noisier
 */

import { describe, expect, it } from 'vitest';
import {
  complement,
  hairpinDensity,
  complementBigramFreq,
  reverseComplementSeq,
  leeDistanceSeq,
  nussinovScore,
  runReduce,
  unpackSymbols,
} from '../src/q2stats.ts';
import { q2EncodeDirect, meanPoolAndNormalise } from '../src/q2.ts';

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
 * Generate a "T4 dialectical" sequence: hairpin signal present but weaker than T3.
 *
 * Models LLM activations on dialectical text: complement palindromes appear at a
 * moderate rate (20%), lower than the T3 dialectical rate (50%), reflecting that
 * general-purpose LLMs have less sharply structured semantic geometry than
 * retrieval-optimised encoders.
 */
function t4DialecticalSeq(length: number, seed: number): number[] {
  const rng = makeRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    if (seq.length + 2 <= length && rng() < 0.2) {
      seq.push(complement(last));
      seq.push(last);
    } else {
      const others = [0, 1, 2, 3].filter((x) => x !== last);
      seq.push(others[Math.floor(rng() * 3)]!);
    }
  }
  return seq.slice(0, length);
}

/**
 * Generate a "T4 direct" sequence: uniform random transitions (null baseline).
 * Same as T1/T3 direct — the null does not depend on the model tier.
 */
function t4DirectSeq(length: number, seed: number): number[] {
  const rng = makeRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last);
    seq.push(others[Math.floor(rng() * 3)]!);
  }
  return seq;
}

/**
 * Generate a "T4 negated" sequence: suppressed hairpins, simulating LLM activations
 * on negated text.  Uses only adjacent (non-complement) transitions.
 */
function t4NegatedSeq(length: number, seed: number): number[] {
  const rng = makeRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
    seq.push(others[Math.floor(rng() * others.length)]!);
  }
  return seq;
}

/**
 * Generate a run-reduced transition sequence from a synthetic random-normal
 * embedding vector; simulates what the Q² pipeline produces from LLM activations.
 */
function encodeSeq(seed: number, n = 128): number[] {
  const rng = makeRng(seed + 11);
  const vec = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u1 = rng() + 1e-10;
    const u2 = rng();
    vec[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  const normed = meanPoolAndNormalise(vec, 1, n);
  const { packed } = q2EncodeDirect(normed, n);
  return runReduce(unpackSymbols(packed, n));
}

// ─── T4-P2: ρ_hp signal in LLM sequences (noisier than T3) ───────────────────

describe('T4-P2: hairpin density signal present in LLM sequences, noisier than T3', () => {
  const SEQ_LENGTH = 600;
  const N_SAMPLES = 60;
  const NULL_RHO = 1 / 9;
  // T4 tolerance is wider than T3 because the signal is noisier
  const TOLERANCE = 0.08;

  it('T4 dialectical sequences have ρ_hp > null baseline (enriched despite noise)', () => {
    const rhops: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      rhops.push(hairpinDensity(t4DialecticalSeq(SEQ_LENGTH, seed)));
    }
    const mean = rhops.reduce((a, b) => a + b, 0) / rhops.length;
    // Signal is weaker than T3 (20% hairpin rate vs 50%), but still above null
    expect(mean).toBeGreaterThan(NULL_RHO);
  });

  it('T4 direct sequences have ρ_hp ≈ 1/9 with wider tolerance than T3', () => {
    const rhops: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      rhops.push(hairpinDensity(t4DirectSeq(SEQ_LENGTH, seed)));
    }
    const mean = rhops.reduce((a, b) => a + b, 0) / rhops.length;
    expect(mean).toBeGreaterThan(NULL_RHO - TOLERANCE);
    expect(mean).toBeLessThan(NULL_RHO + TOLERANCE);
  });

  it('T4 negated sequences have ρ_hp = 0 (no complement palindromes)', () => {
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      expect(hairpinDensity(t4NegatedSeq(SEQ_LENGTH, seed))).toBe(0);
    }
  });

  it('T4 ordering holds: Dialectical > Direct > Negated (on mean)', () => {
    const means = [t4DialecticalSeq, t4DirectSeq, t4NegatedSeq].map((gen) => {
      const rhops: number[] = [];
      for (let seed = 0; seed < N_SAMPLES; seed++) {
        rhops.push(hairpinDensity(gen(SEQ_LENGTH, seed)));
      }
      return rhops.reduce((a, b) => a + b, 0) / rhops.length;
    });
    const [mDial, mDir, mNeg] = means;
    expect(mDial).toBeGreaterThan(mDir!);
    expect(mDir).toBeGreaterThan(mNeg!);
  });

  it('T4 dialectical effect size is smaller than T3 dialectical effect size', () => {
    // T3 dialectical: 50% hairpin rate → mean ρ_hp >> 1/9
    // T4 dialectical: 20% hairpin rate → mean ρ_hp moderately > 1/9
    // Both are above null, but T3 has a larger effect.
    const t3Rho: number[] = [];
    const t4Rho: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      // T3 dialectical (50% rate)
      const t3rng = makeRng(seed);
      const t3Seq: number[] = [Math.floor(t3rng() * 4)];
      while (t3Seq.length < SEQ_LENGTH) {
        const last = t3Seq[t3Seq.length - 1]!;
        if (t3Seq.length + 2 <= SEQ_LENGTH && t3rng() < 0.5) {
          t3Seq.push(complement(last));
          t3Seq.push(last);
        } else {
          const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
          t3Seq.push(others[Math.floor(t3rng() * others.length)]!);
        }
      }
      t3Rho.push(hairpinDensity(t3Seq.slice(0, SEQ_LENGTH)));
      t4Rho.push(hairpinDensity(t4DialecticalSeq(SEQ_LENGTH, seed)));
    }
    const meanT3 = t3Rho.reduce((a, b) => a + b, 0) / t3Rho.length;
    const meanT4 = t4Rho.reduce((a, b) => a + b, 0) / t4Rho.length;
    // T3 has higher hairpin rate → T3 mean should be above T4 mean
    expect(meanT3).toBeGreaterThan(meanT4);
    // Both above null
    expect(meanT4).toBeGreaterThan(NULL_RHO);
  });

  it('T4 dialectical sequences show statistically significant enrichment vs. null', () => {
    // Collect ρ_hp values for T4 dialectical and compare to 1/9 null
    const rhops: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      rhops.push(hairpinDensity(t4DialecticalSeq(SEQ_LENGTH, seed)));
    }
    // Count how many exceed the null
    const aboveNull = rhops.filter((r) => r > NULL_RHO).length;
    // At least 70% of samples should exceed the null (one-sided enrichment)
    expect(aboveNull / N_SAMPLES).toBeGreaterThan(0.7);
  });
});

// ─── T4-P3: Complement-bigram suppression < 1/3 in LLM sequences ─────────────

describe('T4-P3: complement bigram suppression < 1/3 in T4 sequences', () => {
  const NULL_CBF = 1 / 3;
  const SEQ_LENGTH = 600;
  const N_SAMPLES = 50;

  it('T4 negated sequences have complement bigram frequency = 0', () => {
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      expect(complementBigramFreq(t4NegatedSeq(SEQ_LENGTH, seed))).toBe(0);
    }
  });

  it('T4 direct sequences have complement bigram frequency ≈ 1/3 (null baseline)', () => {
    const cbfs: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      cbfs.push(complementBigramFreq(t4DirectSeq(SEQ_LENGTH, seed)));
    }
    const mean = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
    expect(mean).toBeGreaterThan(NULL_CBF - 0.05);
    expect(mean).toBeLessThan(NULL_CBF + 0.05);
  });

  it('T4 negated sequences have complement bigram frequency well below null', () => {
    const cbfs: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      cbfs.push(complementBigramFreq(t4NegatedSeq(SEQ_LENGTH, seed)));
    }
    const mean = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
    expect(mean).toBeLessThan(NULL_CBF - 0.2);
  });

  it('negated sequences have complement bigram frequency well below direct (null) baseline', () => {
    // The P3 prediction is that structured sequences avoiding abrupt semantic
    // transitions suppress complement bigrams.  Negated sequences use only
    // adjacent (Lee-distance-1, non-complement) transitions — the strongest form
    // of complement suppression.
    const cbfsNeg: number[] = [];
    const cbfsDirect: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      cbfsNeg.push(complementBigramFreq(t4NegatedSeq(SEQ_LENGTH, seed)));
      cbfsDirect.push(complementBigramFreq(t4DirectSeq(SEQ_LENGTH, seed)));
    }
    const meanNeg = cbfsNeg.reduce((a, b) => a + b, 0) / cbfsNeg.length;
    const meanDirect = cbfsDirect.reduce((a, b) => a + b, 0) / cbfsDirect.length;
    // Negated sequences suppress complements entirely → lower cbf than random
    expect(meanNeg).toBeLessThan(meanDirect);
  });

  it('synthetic LLM-activation embeddings have cbf consistent with predictions', () => {
    // Encode random normal vectors (simulating LLM activations) and check cbf
    const cbfs: number[] = [];
    for (let seed = 0; seed < 50; seed++) {
      const seq = encodeSeq(seed, 128);
      if (seq.length >= 2) cbfs.push(complementBigramFreq(seq));
    }
    // With random normal activations, cbf should be near null (≈ 1/3)
    const mean = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
    // With random normal activations, cbf should be near the null (≈ 1/3).
    // Use a ±0.1 tolerance around 1/3 ≈ 0.333 to catch meaningful regressions.
    expect(mean).toBeGreaterThan(1 / 3 - 0.1);
    expect(mean).toBeLessThan(1 / 3 + 0.1);
  });
});

// ─── T4-P5: Reverse-complement antonym retrieval (smaller effect than T3) ─────

describe('T4-P5: reverse-complement antonym retrieval — above chance, smaller than T3', () => {
  it('RC(A) is the exact antonym of A (Lee distance 0 from itself)', () => {
    const rng = makeRng(404);
    for (let trial = 0; trial < 30; trial++) {
      const len = 8 + Math.floor(rng() * 12);
      const seq = Array.from({ length: len }, () => Math.floor(rng() * 4));
      const rc = reverseComplementSeq(seq);
      // RC(RC(A)) = A
      expect(reverseComplementSeq(rc)).toEqual(seq);
      // Lee distance from RC(A) to itself = 0
      expect(leeDistanceSeq(rc, rc)).toBe(0);
    }
  });

  it('antonym pair retrieved above chance over a noisy T4 corpus', () => {
    // Model "noisy" T4 antonyms: B ≈ RC(A) with some perturbation.
    // Even a slightly perturbed antonym should be retrieved closer than random.
    const rng = makeRng(808);
    let hits = 0;
    const N_TRIALS = 50;

    for (let trial = 0; trial < N_TRIALS; trial++) {
      const len = 12;
      const A = Array.from({ length: len }, () => Math.floor(rng() * 4));
      const exactAntonym = reverseComplementSeq(A);

      // Perturb the antonym at 1–2 positions (simulating LLM noise)
      const noisyAntonym = [...exactAntonym];
      const noisyPositions = 1 + Math.floor(rng() * 2); // 1 or 2 perturbations
      for (let p = 0; p < noisyPositions; p++) {
        const pos = Math.floor(rng() * len);
        noisyAntonym[pos] = (noisyAntonym[pos]! + 1 + Math.floor(rng() * 3)) % 4;
      }

      // Build a corpus of random distractors + the noisy antonym
      const distractors: number[][] = [];
      for (let d = 0; d < 10; d++) {
        distractors.push(Array.from({ length: len }, () => Math.floor(rng() * 4)));
      }

      const corpus = [noisyAntonym, ...distractors];
      const query = reverseComplementSeq(A);

      // Find nearest neighbor
      let minDist = Infinity;
      let minIdx = -1;
      for (let i = 0; i < corpus.length; i++) {
        const d = leeDistanceSeq(query, corpus[i]!);
        if (d < minDist) { minDist = d; minIdx = i; }
      }

      if (minIdx === 0) hits++; // antonym was retrieved at rank 1
    }

    // Should retrieve the antonym at rank 1 more than chance (> 1/11 ≈ 9%)
    const hitRate = hits / N_TRIALS;
    expect(hitRate).toBeGreaterThan(0.09);
  });

  it('exact antonym is always retrieved at rank 1 (zero noise case)', () => {
    const rng = makeRng(1212);
    for (let trial = 0; trial < 20; trial++) {
      const len = 10;
      const A = Array.from({ length: len }, () => Math.floor(rng() * 4));
      const exactAntonym = reverseComplementSeq(A);

      // Corpus: exact antonym + 10 random distractors
      const corpus = [exactAntonym];
      for (let d = 0; d < 10; d++) {
        corpus.push(Array.from({ length: len }, () => Math.floor(rng() * 4)));
      }

      const query = reverseComplementSeq(A);
      let minDist = Infinity;
      let minIdx = -1;
      for (let i = 0; i < corpus.length; i++) {
        const d = leeDistanceSeq(query, corpus[i]!);
        if (d < minDist) { minDist = d; minIdx = i; }
      }

      // Exact antonym is always retrieved at rank 1 (distance 0)
      expect(minIdx).toBe(0);
      expect(minDist).toBe(0);
    }
  });
});

// ─── T4-P7: Secondary structure complexity (noisier than T3) ─────────────────

describe('T4-P7: secondary structure complexity present but noisier in T4 sequences', () => {
  it('T4 dialectical sequences have higher mean Nussinov score than negated', () => {
    const N = 40;
    const SEQ_LENGTH = 60;
    let totalDial = 0;
    let totalNeg = 0;

    for (let seed = 0; seed < N; seed++) {
      totalDial += nussinovScore(t4DialecticalSeq(SEQ_LENGTH, seed));
      totalNeg += nussinovScore(t4NegatedSeq(SEQ_LENGTH, seed));
    }

    const meanDial = totalDial / N;
    const meanNeg = totalNeg / N;

    // Dialectical sequences contain complement palindromes → Nussinov > 0
    expect(meanDial).toBeGreaterThan(meanNeg);
    // Negated sequences have no complement palindromes → Nussinov = 0
    expect(meanNeg).toBe(0);
  });

  it('T4 dialectical sequences have positive Nussinov score on average', () => {
    const N = 40;
    const SEQ_LENGTH = 50;
    let total = 0;

    for (let seed = 0; seed < N; seed++) {
      total += nussinovScore(t4DialecticalSeq(SEQ_LENGTH, seed));
    }

    const mean = total / N;
    expect(mean).toBeGreaterThan(0);
  });

  it('Nussinov score is non-negative for any transition sequence', () => {
    const rng = makeRng(2525);
    for (let trial = 0; trial < 30; trial++) {
      const len = 5 + Math.floor(rng() * 30);
      const seq = Array.from({ length: len }, () => Math.floor(rng() * 4));
      expect(nussinovScore(seq)).toBeGreaterThanOrEqual(0);
    }
  });

  it('Nussinov score ≤ floor(sequence length / 3) (maximum pair density)', () => {
    // Each complement pair occupies at least 3 symbols (x, θ(x), x), so the maximum
    // number of non-crossing pairs is bounded by floor(n/3).
    const rng = makeRng(3636);
    for (let trial = 0; trial < 20; trial++) {
      const len = 6 + Math.floor(rng() * 30);
      const seq = Array.from({ length: len }, () => Math.floor(rng() * 4));
      const score = nussinovScore(seq);
      expect(score).toBeLessThanOrEqual(Math.floor(len / 3));
    }
  });

  it('T4 has higher coefficient of variation (CV) in Nussinov scores than T3 (noisier signal)', () => {
    // T3 dialectical uses 50% hairpin rate; T4 uses 20%.
    // Lower hairpin rate → weaker signal relative to its own variability, i.e. higher
    // coefficient of variation (CV = std / mean).  The CV is the correct measure of
    // "noisiness" when comparing systems with different mean signal levels.
    const SEQ_LENGTH = 60;
    const N = 50;

    function cv(xs: number[]): number {
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
      if (mean === 0) return Infinity;
      const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
      return Math.sqrt(variance) / mean;
    }

    const t3Scores: number[] = [];
    const t4Scores: number[] = [];
    for (let seed = 0; seed < N; seed++) {
      // T3 dialectical: 50% hairpin rate
      const t3rng = makeRng(seed + 2000);
      const t3Seq: number[] = [Math.floor(t3rng() * 4)];
      while (t3Seq.length < SEQ_LENGTH) {
        const last = t3Seq[t3Seq.length - 1]!;
        if (t3Seq.length + 2 <= SEQ_LENGTH && t3rng() < 0.5) {
          t3Seq.push(complement(last));
          t3Seq.push(last);
        } else {
          const others = [0, 1, 2, 3].filter(
            (x) => x !== last && x !== complement(last),
          );
          t3Seq.push(others[Math.floor(t3rng() * others.length)]!);
        }
      }
      t3Scores.push(nussinovScore(t3Seq.slice(0, SEQ_LENGTH)));
      t4Scores.push(nussinovScore(t4DialecticalSeq(SEQ_LENGTH, seed + 2000)));
    }

    // T3 mean should be higher (stronger signal)
    const meanT3 = t3Scores.reduce((a, b) => a + b, 0) / N;
    const meanT4 = t4Scores.reduce((a, b) => a + b, 0) / N;
    expect(meanT3).toBeGreaterThanOrEqual(meanT4);

    // T4 should have a higher CV (noisier relative to its own mean signal level)
    const cvT3 = cv(t3Scores);
    const cvT4 = cv(t4Scores);
    expect(cvT4).toBeGreaterThan(cvT3);
  });
});
