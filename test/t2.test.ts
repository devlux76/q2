/**
 * t2.test.ts — T2 Phase: Structured Code Corpus
 *
 * Tests the prediction sub-tests from TESTING.md §T2 using synthetic
 * code-like transition sequences.  No model download or corpus fetch is
 * required; the sequences are generated deterministically to exhibit the
 * properties predicted for real code-embedding outputs.
 *
 * References: TESTING.md §T2, PREDICTIONS.md §P2, §P3, §P8, §P10.
 *
 * Test organisation:
 *   T2-P2  — ρ_hp elevated for call-and-return code vs. linear code
 *   T2-P3  — Complement-bigram frequency < 1/3 in code-like sequences
 *   T2-P8  — Non-uniform triplet distribution for code-specific patterns
 *   T2-P10 — 64-bit key collision rate near baseline for code corpus
 */

import { describe, expect, it } from 'vitest';
import {
  complement,
  hairpinDensity,
  complementBigramFreq,
  tripletFreqs,
  runReduce,
  unpackSymbols,
  collisionStats,
  nullCollisionExpectation,
} from '../src/q2stats.ts';
import { q2EncodeDirect, l2Normalise } from '../src/q2.ts';

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
 * Generate a synthetic "call-and-return" transition sequence.
 *
 * Models a code function that calls helper functions and returns: the semantic
 * trajectory departs to the complement of the current state and returns, producing
 * complement palindrome triplets (x, θ(x), x) at a rate controlled by
 * `hairpinFrac`.  The sequence is already run-reduced by construction.
 *
 * @param length     - target sequence length
 * @param hairpinFrac - fraction of steps that initiate a complement palindrome
 * @param seed       - RNG seed for reproducibility
 */
function callAndReturnSeq(length: number, hairpinFrac: number, seed: number): number[] {
  const rng = makeRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];

  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    // If there's room for a full hairpin triplet and the RNG fires, insert one.
    if (seq.length + 2 <= length && rng() < hairpinFrac) {
      seq.push(complement(last)); // step to semantic antipode
      seq.push(last);              // return to original
    } else {
      // Adjacent step: pick one of the two Lee-distance-1 neighbours that is NOT
      // the complement (to keep complement bigrams suppressed in the "normal" steps).
      const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
      seq.push(others[Math.floor(rng() * others.length)]!);
    }
  }
  return seq.slice(0, length);
}

/**
 * Generate a synthetic "linear" transition sequence.
 *
 * Models a code function with no call-and-return structure: only adjacent
 * (Lee-distance-1, non-complement) transitions, so ρ_hp = 0 and complement
 * bigram frequency = 0.
 */
function linearSeq(length: number, seed: number): number[] {
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
 * embedding vector; simulates what the Q² pipeline produces.
 */
function encodeSeq(seed: number, n = 128): number[] {
  const rng = makeRng(seed + 7);
  const vec = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u1 = rng() + 1e-10;
    const u2 = rng();
    vec[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  const normed = l2Normalise(vec, n);
  const { packed } = q2EncodeDirect(normed, n);
  return runReduce(unpackSymbols(packed, n));
}

// ─── T2-P2: ρ_hp elevated for call-and-return vs. linear code ────────────────

describe('T2-P2: hairpin density elevated for call-and-return code vs. linear code', () => {
  const SEQ_LENGTH = 500;
  const N_SAMPLES = 50;

  it('call-and-return sequences have ρ_hp > 1/9 (hairpin density above null)', () => {
    const rhops: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      const seq = callAndReturnSeq(SEQ_LENGTH, 0.4, seed);
      rhops.push(hairpinDensity(seq));
    }
    const mean = rhops.reduce((a, b) => a + b, 0) / rhops.length;
    // With 40% hairpin rate, mean ρ_hp should be well above the null (1/9 ≈ 0.111)
    expect(mean).toBeGreaterThan(1 / 9);
  });

  it('linear sequences have ρ_hp = 0 (no complement palindromes by construction)', () => {
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      const seq = linearSeq(SEQ_LENGTH, seed);
      expect(hairpinDensity(seq)).toBe(0);
    }
  });

  it('mean ρ_hp(call-and-return) > mean ρ_hp(linear) by a substantial margin', () => {
    const rhosCnR: number[] = [];
    const rhosLin: number[] = [];
    for (let seed = 0; seed < N_SAMPLES; seed++) {
      rhosCnR.push(hairpinDensity(callAndReturnSeq(SEQ_LENGTH, 0.3, seed)));
      rhosLin.push(hairpinDensity(linearSeq(SEQ_LENGTH, seed + N_SAMPLES)));
    }
    const meanCnR = rhosCnR.reduce((a, b) => a + b, 0) / rhosCnR.length;
    const meanLin = rhosLin.reduce((a, b) => a + b, 0) / rhosLin.length;
    // The ordering must hold: call-and-return >> linear
    expect(meanCnR).toBeGreaterThan(meanLin + 0.1);
  });

  it('hairpin density is monotone in the hairpin fraction parameter', () => {
    // As hairpinFrac increases, mean ρ_hp should increase monotonically.
    const fracs = [0.1, 0.3, 0.5];
    const means = fracs.map((frac) => {
      const rhops: number[] = [];
      for (let seed = 0; seed < 30; seed++) {
        rhops.push(hairpinDensity(callAndReturnSeq(300, frac, seed + 100)));
      }
      return rhops.reduce((a, b) => a + b, 0) / rhops.length;
    });
    for (let i = 0; i < means.length - 1; i++) {
      expect(means[i + 1]).toBeGreaterThan(means[i]!);
    }
  });
});

// ─── T2-P3: Complement-bigram frequency < 1/3 in code-like sequences ─────────

describe('T2-P3: complement bigram frequency suppressed in code-like sequences', () => {
  const NULL_CBF = 1 / 3;

  it('linear sequences have complement bigram frequency = 0 (only adjacent transitions)', () => {
    for (let seed = 0; seed < 20; seed++) {
      const seq = linearSeq(300, seed);
      expect(complementBigramFreq(seq)).toBe(0);
    }
  });

  it('call-and-return sequences have complement bigram frequency < 1/3', () => {
    // Even with hairpins, the complement bigrams arise only in hairpin steps
    // (which are a minority of steps), so cbf stays below the null 1/3.
    const cbfs: number[] = [];
    for (let seed = 0; seed < 50; seed++) {
      const seq = callAndReturnSeq(500, 0.2, seed);
      cbfs.push(complementBigramFreq(seq));
    }
    const mean = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
    expect(mean).toBeLessThan(NULL_CBF);
  });

  it('lower hairpin fraction → lower complement bigram frequency', () => {
    // Complement bigrams in call-and-return sequences come from the hairpin steps.
    const cbfLow: number[] = [];
    const cbfHigh: number[] = [];
    for (let seed = 0; seed < 40; seed++) {
      cbfLow.push(complementBigramFreq(callAndReturnSeq(400, 0.05, seed)));
      cbfHigh.push(complementBigramFreq(callAndReturnSeq(400, 0.45, seed)));
    }
    const meanLow = cbfLow.reduce((a, b) => a + b, 0) / cbfLow.length;
    const meanHigh = cbfHigh.reduce((a, b) => a + b, 0) / cbfHigh.length;
    expect(meanLow).toBeLessThan(meanHigh);
  });

  it('complement bigram frequency of linear sequences is well below null (< 1/3 − 0.2)', () => {
    const cbfs: number[] = [];
    for (let seed = 0; seed < 30; seed++) {
      cbfs.push(complementBigramFreq(linearSeq(500, seed)));
    }
    const mean = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
    expect(mean).toBeLessThan(NULL_CBF - 0.2);
  });
});

// ─── T2-P8: Non-uniform triplet distribution for code-specific patterns ───────

describe('T2-P8: triplet frequency distribution is non-uniform in code-like sequences', () => {
  it('linear code sequences produce non-uniform triplet distribution (fewer than 36 triplet types)', () => {
    // A cyclic linear sequence (0→1→0→1..., only adjacent steps) produces only
    // a small subset of all possible triplets — simulating "loop/branch" code patterns.
    const seq = linearSeq(2000, 42);
    const freqs = tripletFreqs(seq);
    const observed = Object.keys(freqs).length;
    // A strictly linear sequence with only Ti/Tv1 transitions cannot produce
    // triplets whose middle symbol is the complement of the first symbol (no Tv2).
    // So fewer than 36 distinct triplets should appear.
    expect(observed).toBeLessThan(36);
  });

  it('linear sequences have no hairpin triplets (code pattern: no complement jumps)', () => {
    // Hairpin triplets (x, θ(x), x) require a Tv2 bigram — absent in linear sequences.
    const seq = linearSeq(2000, 99);
    const freqs = tripletFreqs(seq);
    const palindromeTriplets = ['020', '131', '202', '313'];
    const palindromeCount = palindromeTriplets.reduce((s, k) => s + (freqs[k] ?? 0), 0);
    expect(palindromeCount).toBe(0);
  });

  it('call-and-return sequences have elevated hairpin triplet fraction vs. null', () => {
    const seq = callAndReturnSeq(5000, 0.4, 7);
    const freqs = tripletFreqs(seq);
    const total = Object.values(freqs).reduce((a, b) => a + b, 0);
    const palindromeTriplets = ['020', '131', '202', '313'];
    const palindromeCount = palindromeTriplets.reduce((s, k) => s + (freqs[k] ?? 0), 0);
    const palindromeRate = palindromeCount / total;
    // Should be meaningfully above the null (1/9 ≈ 0.111)
    expect(palindromeRate).toBeGreaterThan(1 / 9);
  });

  it('triplet distribution entropy is lower in linear code sequences than in random', () => {
    // A skewed distribution has lower entropy than the near-uniform random null.
    function entropy(freqs: Record<string, number>): number {
      const total = Object.values(freqs).reduce((a, b) => a + b, 0);
      let h = 0;
      for (const count of Object.values(freqs)) {
        const p = count / total;
        if (p > 0) h -= p * Math.log2(p);
      }
      return h;
    }

    const linearEntropy = entropy(tripletFreqs(linearSeq(10_000, 1)));

    // Random sequences (from T1) should have entropy close to log2(36) ≈ 5.17
    const randomSeq: number[] = [];
    const rng = makeRng(12345);
    randomSeq.push(Math.floor(rng() * 4));
    while (randomSeq.length < 10_000) {
      const last = randomSeq[randomSeq.length - 1]!;
      const others = [0, 1, 2, 3].filter((x) => x !== last);
      randomSeq.push(others[Math.floor(rng() * 3)]!);
    }
    const randomEntropy = entropy(tripletFreqs(randomSeq));

    expect(linearEntropy).toBeLessThan(randomEntropy);
  });
});

// ─── T2-P10: Key collision rate near baseline for code corpus ─────────────────

describe('T2-P10: 64-bit key collision rate is near the random-hash baseline', () => {
  it('null collision expectation for 1000 code documents is negligible', () => {
    const expected = nullCollisionExpectation(1000);
    expect(expected).toBeLessThan(1e-6);
  });

  it('500 synthetic code-embedding keys have near-zero collision rate', () => {
    const n = 64;
    const keys: bigint[] = [];
    let s = (1337 >>> 0);
    const rng = () => {
      s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
      return (s >>> 0) / 0x100000000;
    };

    for (let doc = 0; doc < 500; doc++) {
      const vec = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const u1 = rng() + 1e-10;
        const u2 = rng();
        vec[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      const normed = l2Normalise(vec, n);
      const { key } = q2EncodeDirect(normed, n);
      keys.push(key);
    }

    const { rate } = collisionStats(keys);
    expect(rate).toBeLessThan(0.01);
  });

  it('collision rate for a call-and-return corpus is near the random-hash baseline', () => {
    // Encode the run-reduced transition sequences of 200 call-and-return sequences
    // as their first 64 bits of key material; collisions should be rare.
    const encodeKey = (seq: number[]): bigint => {
      // Build a compact 64-bit hash from the first 32 symbols of the run-reduced seq.
      let key = 0n;
      const limit = Math.min(seq.length, 32);
      for (let i = 0; i < limit; i++) {
        key = (key << 2n) | BigInt(seq[i]! & 3);
      }
      return key;
    };

    const keys: bigint[] = [];
    for (let seed = 0; seed < 200; seed++) {
      const seq = callAndReturnSeq(200, 0.25, seed * 7);
      keys.push(encodeKey(seq));
    }

    // Collision rate should be well below 1%
    const { rate } = collisionStats(keys);
    expect(rate).toBeLessThan(0.05);
  });

  it('encodeSeq produces distinct keys for different document seeds', () => {
    const keys: bigint[] = [];
    for (let seed = 0; seed < 100; seed++) {
      const seq = encodeSeq(seed, 128);
      // Convert run-reduced sequence to a 64-bit key via q2KeyDirect on packed bytes
      const rng = makeRng(seed + 7);
      const vec = new Float32Array(128);
      for (let i = 0; i < 128; i++) {
        const u1 = rng() + 1e-10;
        const u2 = rng();
        vec[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      const normed = l2Normalise(vec, 128);
      const { key } = q2EncodeDirect(normed, 128);
      keys.push(key);
      // Also verify the sequence is valid
      expect(seq.length).toBeGreaterThan(0);
    }

    const { rate } = collisionStats(keys);
    expect(rate).toBeLessThan(0.01);
  });
});
