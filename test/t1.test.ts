/**
 * t1.test.ts — T1 Phase: Random Text Null Distribution
 *
 * Establishes the null distributions for all statistics measured in later phases
 * (T2–T4).  These tests use synthetic randomly-generated embeddings — no model
 * download is needed.  They verify that the Q² analysis functions produce the
 * theoretically expected null values when applied to uniformly random transition
 * sequences.
 *
 * References: TESTING.md §T1, PREDICTIONS.md §P2, §P3, §P8.
 *
 * Test organisation:
 *   T1-P2  — ρ_hp null baseline ≈ 1/9
 *   T1-P3  — Complement bigram frequency null baseline ≈ 1/3
 *   T1-P8  — Triplet frequency null baseline ≈ uniform
 */

import { describe, expect, it } from 'vitest';
import {
  complement,
  hairpinDensity,
  complementBigramFreq,
  tripletFreqs,
  runReduce,
  unpackSymbols,
} from '../src/q2stats.ts';
import { q2EncodeDirect, meanPoolAndNormalise } from '../src/q2.ts';

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Seeded xorshift32 RNG — produces uniform unsigned 32-bit integers with
 * full period 2^32 − 1.  Lower bits are well-distributed (unlike a simple
 * LCG, whose lower bits have much shorter periods).
 */
function makeRng(seed: number) {
  let s = (seed === 0 ? 123456789 : seed) >>> 0;
  return function () {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000; // uniform float in [0, 1)
  };
}

/**
 * Generate a pseudo-random run-reduced transition sequence of exactly
 * `targetLength` elements.  Each next symbol is sampled uniformly from the
 * 3 non-current symbols — this is the Markov chain that produces the null
 * distribution described in PREDICTIONS.md (ρ_hp → 1/9, cbf → 1/3).
 */
function randomTransitionSeq(targetLength: number, seed: number): number[] {
  const rng = makeRng(seed);
  const seq: number[] = [];
  // First symbol: uniform over {0,1,2,3}
  seq.push(Math.floor(rng() * 4));
  while (seq.length < targetLength) {
    const last = seq[seq.length - 1]!;
    // Sample uniformly from the 3 non-last symbols
    const others = [0, 1, 2, 3].filter((x) => x !== last);
    seq.push(others[Math.floor(rng() * 3)]!);
  }
  return seq;
}

/**
 * Generate a synthetic embedding vector, encode with Q², and return the
 * run-reduced transition sequence.
 */
function encodeSeq(seed: number, n = 128): number[] {
  const rng = makeRng(seed + 1); // +1 so seed=0 still works
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

// ─── T1-P2: Hairpin density null baseline ────────────────────────────────────

describe('T1-P2: ρ_hp null baseline ≈ 1/9 for uniformly random sequences', () => {
  const NULL_RHO_HP = 1 / 9;
  const TOLERANCE = 0.04; // ±4 percentage points; tightens with more samples

  it('ρ_hp ≈ 1/9 for a long uniform random transition sequence', () => {
    // Generate a 10 000-element run-reduced sequence; null expectation is exactly 1/9.
    const seq = randomTransitionSeq(10_000, 1234);
    const rho = hairpinDensity(seq);
    expect(rho).toBeGreaterThan(NULL_RHO_HP - TOLERANCE);
    expect(rho).toBeLessThan(NULL_RHO_HP + TOLERANCE);
  });

  it('ρ_hp ≈ 1/9 for synthetic random-normal embeddings (mean over corpus)', () => {
    // Encode 200 synthetic random embeddings and compute mean ρ_hp.
    const rhops: number[] = [];
    for (let seed = 0; seed < 200; seed++) {
      const seq = encodeSeq(seed + 1);
      const rho = hairpinDensity(seq);
      if (seq.length >= 3) rhops.push(rho);
    }
    const mean = rhops.reduce((a, b) => a + b, 0) / rhops.length;
    // Mean should be close to 1/9 ≈ 0.111 within ±5 percentage points.
    expect(mean).toBeGreaterThan(NULL_RHO_HP - 0.05);
    expect(mean).toBeLessThan(NULL_RHO_HP + 0.05);
  });

  it('hairpin density is stable across document lengths', () => {
    // Null expectation does not depend on sequence length.
    const lengths = [100, 500, 2000];
    for (const length of lengths) {
      const seq = randomTransitionSeq(length, 42);
      const rho = hairpinDensity(seq);
      expect(rho).toBeGreaterThan(NULL_RHO_HP - TOLERANCE);
      expect(rho).toBeLessThan(NULL_RHO_HP + TOLERANCE);
    }
  });
});

// ─── T1-P3: Complement bigram frequency null baseline ────────────────────────

describe('T1-P3: Complement bigram null baseline ≈ 1/3 for uniform random sequences', () => {
  const NULL_CBF = 1 / 3;
  const TOLERANCE = 0.04;

  it('complement bigram frequency ≈ 1/3 for a long uniform random sequence', () => {
    const seq = randomTransitionSeq(10_000, 5678);
    const cbf = complementBigramFreq(seq);
    expect(cbf).toBeGreaterThan(NULL_CBF - TOLERANCE);
    expect(cbf).toBeLessThan(NULL_CBF + TOLERANCE);
  });

  it('complement bigram frequency ≈ 1/3 for synthetic random-normal embeddings', () => {
    const cbfs: number[] = [];
    for (let seed = 0; seed < 200; seed++) {
      const seq = encodeSeq(seed + 500);
      if (seq.length >= 2) cbfs.push(complementBigramFreq(seq));
    }
    const mean = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
    expect(mean).toBeGreaterThan(NULL_CBF - 0.05);
    expect(mean).toBeLessThan(NULL_CBF + 0.05);
  });

  it('complement bigrams use both complement pairs (0↔2 and 1↔3)', () => {
    // Generate a long sequence and verify complement bigrams appear in both families.
    const seq = randomTransitionSeq(5_000, 9999);
    let gc = 0; // 0→2 or 2→0
    let at = 0; // 1→3 or 3→1
    for (let i = 0; i < seq.length - 1; i++) {
      const a = seq[i]!;
      const b = seq[i + 1]!;
      if (b === complement(a)) {
        if (a === 0 || a === 2) gc++;
        else at++;
      }
    }
    // Both complement families should be represented
    expect(gc).toBeGreaterThan(0);
    expect(at).toBeGreaterThan(0);
    // They should be roughly equal in a uniform random sequence
    const ratio = gc / (gc + at);
    expect(ratio).toBeGreaterThan(0.3);
    expect(ratio).toBeLessThan(0.7);
  });
});

// ─── T1-P8: Triplet frequency null baseline ───────────────────────────────────

describe('T1-P8: Triplet frequency null baseline ≈ uniform for random sequences', () => {
  it('all valid triplet forms are observed in a long uniform random sequence', () => {
    // A run-reduced sequence has r_i ≠ r_{i+1}; for triplets (x,y,z), y ≠ x
    // and z ≠ y (but z may equal x).  There are 4×3×3=36 valid triplet forms.
    const seq = randomTransitionSeq(20_000, 7777);
    const freqs = tripletFreqs(seq);
    // Count distinct triplets observed
    const observed = Object.keys(freqs).length;
    // Should observe close to 36 = 4×3×3 valid triplets
    expect(observed).toBeGreaterThan(30);
  });

  it('no single triplet accounts for more than 20% of all triplets (not degenerate)', () => {
    const seq = randomTransitionSeq(10_000, 1111);
    const freqs = tripletFreqs(seq);
    const total = Object.values(freqs).reduce((a, b) => a + b, 0);
    for (const count of Object.values(freqs)) {
      expect(count / total).toBeLessThan(0.2);
    }
  });

  it('triplet entropy is close to maximum for uniform random sequences', () => {
    // Maximum entropy for 36 equally likely triplets ≈ log2(36) ≈ 5.17 bits.
    // A uniform distribution should be close to maximum; 70% of max is a loose bound.
    const seq = randomTransitionSeq(20_000, 3333);
    const freqs = tripletFreqs(seq);
    const total = Object.values(freqs).reduce((a, b) => a + b, 0);
    let entropy = 0;
    for (const count of Object.values(freqs)) {
      const p = count / total;
      if (p > 0) entropy -= p * Math.log2(p);
    }
    const maxEntropy = Math.log2(36);
    expect(entropy).toBeGreaterThan(0.7 * maxEntropy);
  });

  it('hairpin triplets occur at rate ≈ 1/9 in null', () => {
    // In a random sequence, triplets of the form (x, θ(x), x) should appear
    // with frequency ≈ 1/9, consistent with the ρ_hp null baseline.
    const seq = randomTransitionSeq(10_000, 2222);
    const freqs = tripletFreqs(seq);
    const total = Object.values(freqs).reduce((a, b) => a + b, 0);

    // Complement palindrome triplets: (0,2,0), (1,3,1), (2,0,2), (3,1,3)
    const palindromeTriplets = ['020', '131', '202', '313'];
    const palindromeCount = palindromeTriplets.reduce(
      (s, k) => s + (freqs[k] ?? 0), 0,
    );
    const palindromeRate = palindromeCount / total;

    // Null expectation is 4/36 = 1/9 ≈ 0.111
    expect(palindromeRate).toBeGreaterThan(1 / 9 - 0.04);
    expect(palindromeRate).toBeLessThan(1 / 9 + 0.04);
  });
});

// ─── T1: Transition sequence structure ───────────────────────────────────────

describe('T1: Transition sequence structural properties', () => {
  it('run-reduced sequence from real embedding has no consecutive duplicates', () => {
    for (let seed = 0; seed < 20; seed++) {
      const seq = encodeSeq(seed);
      for (let i = 0; i < seq.length - 1; i++) {
        expect(seq[i]).not.toBe(seq[i + 1]);
      }
    }
  });

  it('run-reduced sequence length is at most n for n-dimensional embedding', () => {
    const n = 128;
    for (let seed = 0; seed < 20; seed++) {
      const seq = encodeSeq(seed, n);
      // run-reduced length ≤ n (in the extreme case every symbol changes)
      expect(seq.length).toBeLessThanOrEqual(n);
      expect(seq.length).toBeGreaterThan(0);
    }
  });

  it('all symbols in run-reduced sequence are valid Z₄ symbols', () => {
    for (let seed = 0; seed < 20; seed++) {
      const seq = encodeSeq(seed);
      for (const s of seq) {
        expect(s).toBeGreaterThanOrEqual(0);
        expect(s).toBeLessThanOrEqual(3);
      }
    }
  });
});
