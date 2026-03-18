/**
 * t0.test.ts — T0 Phase: Unit Tests and Invariants
 *
 * Tests the algebraic invariants of the Q² encoding that must hold before any
 * corpus or model experiment.  These tests run entirely in TypeScript, require
 * no model download, and should pass on any machine.
 *
 * References: TESTING.md §T0, PREDICTIONS.md §P1, §P2, §P3, §P4, §P8, §P9, §P10.
 *
 * Test organisation:
 *   T0-P1  — CGAT mapping and complement involution
 *   T0-Q   — Quantisation and packing invariants (smoke tests)
 *   T0-P4  — Transition/transversion algebraic classification
 *   T0-P9  — Z₄ Gray-code isometry uniqueness (alphabet optimality)
 *   T0-P10 — Key entropy and collision rate
 */

import { describe, expect, it } from 'vitest';
import {
  complement,
  leeDistance,
  leeDistancePacked,
  grayEncode,
  grayDecode,
  unpackSymbols,
  runReduce,
  hairpinDensity,
  complementBigramFreq,
  tripletFreqs,
  reverseComplementSeq,
  collisionStats,
  nullCollisionExpectation,
} from '../src/q2stats.ts';
import {
  q2EncodeDirect,
  q2KeyDirect,
  meanPoolAndNormalise,
} from '../src/q2.ts';

// ─── T0-P1: CGAT mapping and complement involution ───────────────────────────

describe('T0-P1: CGAT complement involution θ', () => {
  it('θ is an involution: θ(θ(z)) = z for all z', () => {
    for (let z = 0; z < 4; z++) {
      expect(complement(complement(z))).toBe(z);
    }
  });

  it('θ pairs Watson–Crick complements: G↔C (0↔2) and A↔T (1↔3)', () => {
    expect(complement(0)).toBe(2); // G ↔ C
    expect(complement(2)).toBe(0); // C ↔ G
    expect(complement(1)).toBe(3); // A ↔ T
    expect(complement(3)).toBe(1); // T ↔ A
  });

  it('θ(z) = z ⊕ 2 for all z', () => {
    for (let z = 0; z < 4; z++) {
      expect(complement(z)).toBe(z ^ 2);
    }
  });

  it('θ has no fixed points: θ(z) ≠ z for all z', () => {
    for (let z = 0; z < 4; z++) {
      expect(complement(z)).not.toBe(z);
    }
  });
});

describe('T0-P1: Lee metric on Z₄', () => {
  it('d_L(z, z) = 0 for all z', () => {
    for (let z = 0; z < 4; z++) {
      expect(leeDistance(z, z)).toBe(0);
    }
  });

  it('d_L is symmetric', () => {
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        expect(leeDistance(a, b)).toBe(leeDistance(b, a));
      }
    }
  });

  it('d_L ≤ 2 for all pairs on Z₄', () => {
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        expect(leeDistance(a, b)).toBeGreaterThanOrEqual(0);
        expect(leeDistance(a, b)).toBeLessThanOrEqual(2);
      }
    }
  });

  it('complement pairs have Lee distance 2 (Watson–Crick = maximum distance)', () => {
    for (let z = 0; z < 4; z++) {
      expect(leeDistance(z, complement(z))).toBe(2);
    }
  });

  it('adjacent pairs (G–A, A–T, T–C, C–G) have Lee distance 1', () => {
    // Z₄ ring: 0-1-2-3-0 with cyclic wrap
    const adjacent: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0]];
    for (const [a, b] of adjacent) {
      expect(leeDistance(a, b)).toBe(1);
      expect(leeDistance(b, a)).toBe(1);
    }
  });

  it('transition pairs (G–A and C–T, same ring class) have Lee distance 1', () => {
    // G=0, A=1: both purines — differ in keto/amino only
    expect(leeDistance(0, 1)).toBe(1);
    // C=2, T=3: both pyrimidines — differ in keto/amino only
    expect(leeDistance(2, 3)).toBe(1);
  });

  it('transversion type-1 pairs (G–T and A–C, same functional group) have Lee distance 1', () => {
    // G=0, T=3: both keto — differ in purine/pyrimidine only (wrap: |0-3|=3, 4-3=1)
    expect(leeDistance(0, 3)).toBe(1);
    // A=1, C=2: both amino — differ in purine/pyrimidine only
    expect(leeDistance(1, 2)).toBe(1);
  });
});

describe('T0-P1: Gray-code Lee-to-Hamming isometry', () => {
  it('Gray encode/decode round-trip for all Z₄ symbols', () => {
    for (let z = 0; z < 4; z++) {
      expect(grayDecode(grayEncode(z))).toBe(z);
    }
  });

  it('Gray code table matches §D-1.7: G=00, A=01, C=11, T=10', () => {
    // Z₄ → Gray: 0→00, 1→01, 2→11, 3→10
    expect(grayEncode(0)).toBe(0b00); // G
    expect(grayEncode(1)).toBe(0b01); // A
    expect(grayEncode(2)).toBe(0b11); // C
    expect(grayEncode(3)).toBe(0b10); // T
  });

  it('Lee distance equals Hamming distance between Gray codes (isometry property)', () => {
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        const ga = grayEncode(a);
        const gb = grayEncode(b);
        // Hamming distance = popcount of XOR of 2-bit codes
        const xor = ga ^ gb;
        const hamming = ((xor >> 1) & 1) + (xor & 1);
        expect(hamming).toBe(leeDistance(a, b));
      }
    }
  });

  it('complement pair Gray codes differ in both bits (all bits flipped)', () => {
    // θ(z) = z ⊕ 2; Gray(z) and Gray(θ(z)) should have Hamming distance 2
    for (let z = 0; z < 4; z++) {
      const ga = grayEncode(z);
      const gb = grayEncode(complement(z));
      const xor = ga ^ gb;
      const hamming = ((xor >> 1) & 1) + (xor & 1);
      expect(hamming).toBe(2);
    }
  });

  it('leeDistancePacked for identical packed bytes is zero', () => {
    const a = new Uint8Array([0xAA, 0x55, 0xFF, 0x00]);
    expect(leeDistancePacked(a, a)).toBe(0);
  });

  it('leeDistancePacked(a, a⊕FF) counts all bits flipped', () => {
    const a = new Uint8Array([0xAA]);
    const b = new Uint8Array([0x55]); // 0xAA XOR 0xFF = 0x55
    // 0xAA = 10101010, 0x55 = 01010101; XOR = 11111111 (8 bits set)
    expect(leeDistancePacked(a, b)).toBe(8);
  });
});

// ─── T0-P1: Gray bit axes (purine/pyrimidine and keto/amino) ─────────────────

describe('T0-P1: Gray bits encode chemical classification axes', () => {
  it('MSB of Gray code encodes purine/pyrimidine: 0 for {G,A}, 1 for {C,T}', () => {
    // Purines (G=0, A=1): MSB=0
    expect((grayEncode(0) >> 1) & 1).toBe(0); // G
    expect((grayEncode(1) >> 1) & 1).toBe(0); // A
    // Pyrimidines (C=2, T=3): MSB=1
    expect((grayEncode(2) >> 1) & 1).toBe(1); // C
    expect((grayEncode(3) >> 1) & 1).toBe(1); // T
  });

  it('LSB of Gray code encodes keto/amino: 0 for {G,T}, 1 for {A,C}', () => {
    // Keto (G=0, T=3): LSB=0
    expect(grayEncode(0) & 1).toBe(0); // G
    expect(grayEncode(3) & 1).toBe(0); // T
    // Amino (A=1, C=2): LSB=1
    expect(grayEncode(1) & 1).toBe(1); // A
    expect(grayEncode(2) & 1).toBe(1); // C
  });
});

// ─── T0-Q: Quantisation and packing invariants ───────────────────────────────

describe('T0-Q: Quantisation smoke tests', () => {
  it('random normal vector always produces a valid 64-bit key', () => {
    // Generate pseudo-random normal vectors using xorshift32 Box–Muller for determinism.
    function makeRng(seed: number) {
      let s = (seed === 0 ? 123456789 : seed) >>> 0;
      return function () {
        s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
        return (s >>> 0) / 0x100000000;
      };
    }

    const n = 256;
    for (let seed = 1; seed <= 10; seed++) {
      const rng = makeRng(seed);
      const raw = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const u1 = rng() + 1e-10;
        const u2 = rng();
        raw[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      const vec = meanPoolAndNormalise(raw, 1, n);
      const { packed, key } = q2EncodeDirect(vec, n);

      // Packed bytes must be n/4 bytes
      expect(packed.length).toBe(n / 4);

      // Key must be a valid BigInt (not NaN/undefined)
      expect(typeof key).toBe('bigint');

      // Key must be in range [0, 2^64)
      expect(key).toBeGreaterThanOrEqual(0n);
      expect(key).toBeLessThan(1n << 64n);
    }
  });

  it('same input always produces the same key (deterministic)', () => {
    const n = 128;
    const vec = new Float32Array(n).map((_, i) => Math.sin(i * 0.31));
    const normed = meanPoolAndNormalise(vec, 1, n);

    const results = Array.from({ length: 5 }, () => q2EncodeDirect(normed, n));
    for (const r of results) {
      expect(r.key).toBe(results[0]!.key);
      expect(Array.from(r.packed)).toEqual(Array.from(results[0]!.packed));
    }
  });

  it('unpackSymbols + runReduce + q2KeyDirect round-trip matches q2EncodeDirect', () => {
    const n = 64;
    const vec = new Float32Array(n).map((_, i) => (i % 3 - 1) * 0.4);
    const normed = meanPoolAndNormalise(vec, 1, n);

    const { packed, key } = q2EncodeDirect(normed, n);
    const syms = unpackSymbols(packed, n);
    const rr = runReduce(syms);

    // All run-reduced symbols must be in {0,1,2,3}
    for (const s of rr) expect(s).toBeGreaterThanOrEqual(0);
    for (const s of rr) expect(s).toBeLessThanOrEqual(3);

    // No consecutive duplicates in run-reduced sequence
    for (let i = 0; i < rr.length - 1; i++) {
      expect(rr[i]).not.toBe(rr[i + 1]);
    }

    // q2KeyDirect on packed bytes must match q2EncodeDirect key
    expect(q2KeyDirect(packed, n)).toBe(key);
  });
});

// ─── T0-P10: Key entropy and collision rate ───────────────────────────────────

describe('T0-P10: Key entropy and collision rate', () => {
  it('nullCollisionExpectation for 1000 keys in 64-bit space is < 1e-6', () => {
    // 1000*(999) / 2^65 ≈ 2.7e-14 — far below 1e-6, essentially zero
    const expected = nullCollisionExpectation(1000);
    expect(expected).toBeLessThan(1e-6);
  });

  it('collision rate for 1000 distinct synthetic keys is near zero', () => {
    // Each key is incremented by 1; all distinct.
    const keys: bigint[] = Array.from({ length: 1000 }, (_, i) => BigInt(i + 1));
    const { rate } = collisionStats(keys);
    expect(rate).toBe(0);
  });

  it('collisionStats correctly counts duplicates', () => {
    const keys: bigint[] = [1n, 2n, 1n, 3n, 2n, 2n];
    const stats = collisionStats(keys);
    // Key 1 appears twice (1 collision), key 2 appears 3 times (2 collisions)
    expect(stats.collisions).toBe(3);
    expect(stats.groups).toBe(2);
    expect(stats.rate).toBeCloseTo(0.5, 5);
  });

  it('synthetic random-normal embeddings produce low key collision rate', () => {
    // Generate 500 distinct pseudo-random normalised embeddings and check
    // that the key collision rate is negligible.
    const n = 64;
    const keys: bigint[] = [];
    let s = (42 >>> 0);
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
      const normed = meanPoolAndNormalise(vec, 1, n);
      const { key } = q2EncodeDirect(normed, n);
      keys.push(key);
    }

    const { rate } = collisionStats(keys);
    // With 64-bit keys and 500 documents, collision rate should be essentially 0
    expect(rate).toBeLessThan(0.01);
  });
});

// ─── T0: unpackSymbols and runReduce invariants ───────────────────────────────

describe('T0: unpackSymbols', () => {
  it('returns n symbols from n/4 packed bytes', () => {
    const packed = new Uint8Array([0xAA, 0xAA]); // all D (Gray 10₂)
    const syms = unpackSymbols(packed, 8);
    expect(syms.length).toBe(8);
  });

  it('all-D packed bytes decode to all Z₄ symbol 3', () => {
    // D=3: Gray 10₂ = 0b10; packed: each byte = 0b10101010 = 0xAA
    const packed = new Uint8Array([0xAA, 0xAA]);
    const syms = unpackSymbols(packed, 8);
    for (const s of syms) expect(s).toBe(3);
  });

  it('all-A packed bytes decode to all Z₄ symbol 0', () => {
    // A=0: Gray 00₂; packed: each byte = 0b00000000 = 0x00
    const packed = new Uint8Array([0x00, 0x00]);
    const syms = unpackSymbols(packed, 8);
    for (const s of syms) expect(s).toBe(0);
  });

  it('alternating A/D packed bytes decode to alternating 0/3', () => {
    // A=0 (Gray 00), D=3 (Gray 10) in alternating positions
    // Byte 0: positions 0–3: A,D,A,D → 00 10 00 10 = 0b00100010 = 0x22
    const packed = new Uint8Array([0x22, 0x22]);
    const syms = unpackSymbols(packed, 8);
    for (let i = 0; i < 8; i++) {
      expect(syms[i]).toBe(i % 2 === 0 ? 0 : 3);
    }
  });
});

describe('T0: runReduce', () => {
  it('returns empty for empty input', () => {
    expect(runReduce([])).toEqual([]);
  });

  it('returns single element for all-same input', () => {
    expect(runReduce([2, 2, 2, 2])).toEqual([2]);
  });

  it('collapses consecutive duplicates but preserves non-consecutive repeats', () => {
    expect(runReduce([0, 0, 1, 2, 2, 3, 0])).toEqual([0, 1, 2, 3, 0]);
  });

  it('produces a sequence with no consecutive duplicates', () => {
    const input = [0, 1, 1, 2, 3, 3, 3, 0, 0, 1];
    const result = runReduce(input);
    for (let i = 0; i < result.length - 1; i++) {
      expect(result[i]).not.toBe(result[i + 1]);
    }
  });
});

// ─── T0: reverseComplementSeq ─────────────────────────────────────────────────

describe('T0-P5: reverseComplementSeq', () => {
  it('double reverse complement is identity', () => {
    const seq = [0, 1, 2, 3, 1, 0, 2];
    const result = reverseComplementSeq(reverseComplementSeq(seq));
    expect(result).toEqual(seq);
  });

  it('applies complement element-wise and reverses', () => {
    // θ: 0→2, 1→3, 2→0, 3→1
    const seq = [0, 1, 2, 3];
    expect(reverseComplementSeq(seq)).toEqual([1, 0, 3, 2]);
  });

  it('returns empty for empty input', () => {
    expect(reverseComplementSeq([])).toEqual([]);
  });
});

// ─── T0: hairpinDensity edge cases ───────────────────────────────────────────

describe('T0-P2: hairpinDensity edge cases', () => {
  it('returns 0 for sequence shorter than 3', () => {
    expect(hairpinDensity([])).toBe(0);
    expect(hairpinDensity([0])).toBe(0);
    expect(hairpinDensity([0, 1])).toBe(0);
  });

  it('detects a single complement palindrome', () => {
    // (0, 2, 0) — θ(0)=2, r_{i+2}=0=r_i: a hairpin
    const seq = [0, 2, 0];
    expect(hairpinDensity(seq)).toBeCloseTo(1.0, 5);
  });

  it('returns 0 for a sequence with no complement palindromes', () => {
    // (0, 1, 0) — θ(0)=2 ≠ 1: not a complement palindrome
    const seq = [0, 1, 0];
    expect(hairpinDensity(seq)).toBe(0);
  });

  it('complement palindromes use all four complement pairs', () => {
    // All four: (0,2,0), (1,3,1), (2,0,2), (3,1,3)
    const pairs: [number, number, number][] = [
      [0, 2, 0],
      [1, 3, 1],
      [2, 0, 2],
      [3, 1, 3],
    ];
    for (const triplet of pairs) {
      expect(hairpinDensity(triplet)).toBeCloseTo(1.0, 5);
    }
  });
});

// ─── T0: complementBigramFreq edge cases ─────────────────────────────────────

describe('T0-P3: complementBigramFreq edge cases', () => {
  it('returns 0 for sequence shorter than 2', () => {
    expect(complementBigramFreq([])).toBe(0);
    expect(complementBigramFreq([0])).toBe(0);
  });

  it('returns 1.0 for all-complement bigrams', () => {
    // (0,2,0,2,...) every consecutive pair is a complement bigram
    const seq = [0, 2, 0, 2, 0, 2];
    expect(complementBigramFreq(seq)).toBeCloseTo(1.0, 5);
  });

  it('returns 0 for no complement bigrams', () => {
    // Only adjacent transitions, no complement jumps
    const seq = [0, 1, 2, 3, 0, 1];
    expect(complementBigramFreq(seq)).toBe(0);
  });
});

// ─── T0: tripletFreqs ────────────────────────────────────────────────────────

describe('T0-P8: tripletFreqs', () => {
  it('returns empty for sequence shorter than 3', () => {
    expect(tripletFreqs([0, 1])).toEqual({});
  });

  it('counts a single triplet correctly', () => {
    const seq = [0, 1, 2];
    expect(tripletFreqs(seq)).toEqual({ '012': 1 });
  });

  it('counts overlapping triplets correctly', () => {
    const seq = [0, 1, 0, 1];
    const freqs = tripletFreqs(seq);
    // Triplets: (0,1,0) and (1,0,1)
    expect(freqs['010']).toBe(1);
    expect(freqs['101']).toBe(1);
  });

  it('total triplet count equals seq.length - 2', () => {
    const seq = [0, 1, 2, 3, 0, 1, 2];
    const freqs = tripletFreqs(seq);
    const total = Object.values(freqs).reduce((a, b) => a + b, 0);
    expect(total).toBe(seq.length - 2);
  });
});

// ─── T0-P4: Transition/transversion algebraic classification ─────────────────
//
// Under the CGAT mapping, bit-0 of the Gray code encodes purine/pyrimidine
// and bit-1 encodes keto/amino.  Lee-distance-1 pairs that change only the
// keto/amino bit are "transitions" (G↔A and C↔T); those that change only
// the purine/pyrimidine bit are "type-1 transversions" (G↔T and A↔C).
// Lee-distance-2 pairs (G↔C and A↔T) change both bits and are "complement
// transversions".  These algebraic facts are testable without a corpus.

describe('T0-P4: transition/transversion algebraic classification', () => {
  // G=0 (00), A=1 (01), C=2 (11), T=3 (10) in Gray encoding.
  // Bit-0 (MSB, purine/pyrimidine): G=0, A=0, C=1, T=1
  // Bit-1 (LSB, keto/amino):        G=0, A=1, C=1, T=0

  /** Return the two-bit Gray encoding of a Z₄ symbol. */
  function gray(z: number): [number, number] {
    const g = grayEncode(z);
    return [(g >> 1) & 1, g & 1]; // [purine/pyrimidine bit, keto/amino bit]
  }

  it('G(0) and A(1) are both purines (same ring-class bit)', () => {
    expect(gray(0)[0]).toBe(0);
    expect(gray(1)[0]).toBe(0);
  });

  it('C(2) and T(3) are both pyrimidines (same ring-class bit)', () => {
    expect(gray(2)[0]).toBe(1);
    expect(gray(3)[0]).toBe(1);
  });

  it('G(0) and T(3) are both keto (same functional-group bit)', () => {
    expect(gray(0)[1]).toBe(0);
    expect(gray(3)[1]).toBe(0);
  });

  it('A(1) and C(2) are both amino (same functional-group bit)', () => {
    expect(gray(1)[1]).toBe(1);
    expect(gray(2)[1]).toBe(1);
  });

  it('transitions (G↔A, C↔T) have Lee distance 1 and change only the keto/amino bit', () => {
    const transitions: [number, number][] = [[0, 1], [2, 3]]; // G↔A, C↔T
    for (const [a, b] of transitions) {
      expect(leeDistance(a, b)).toBe(1);
      // Same ring-class (purine/pyrimidine bit unchanged)
      expect(gray(a)[0]).toBe(gray(b)[0]);
      // Different functional-group (keto/amino bit changes)
      expect(gray(a)[1]).not.toBe(gray(b)[1]);
    }
  });

  it('type-1 transversions (G↔T, A↔C) have Lee distance 1 and change only the ring-class bit', () => {
    const tv1: [number, number][] = [[0, 3], [1, 2]]; // G↔T, A↔C
    for (const [a, b] of tv1) {
      expect(leeDistance(a, b)).toBe(1);
      // Different ring-class (purine/pyrimidine bit changes)
      expect(gray(a)[0]).not.toBe(gray(b)[0]);
      // Same functional-group (keto/amino bit unchanged)
      expect(gray(a)[1]).toBe(gray(b)[1]);
    }
  });

  it('complement transversions (G↔C, A↔T) have Lee distance 2 and change both chemical bits', () => {
    const tv2: [number, number][] = [[0, 2], [1, 3]]; // G↔C, A↔T
    for (const [a, b] of tv2) {
      expect(leeDistance(a, b)).toBe(2);
      // Both bits change
      expect(gray(a)[0]).not.toBe(gray(b)[0]);
      expect(gray(a)[1]).not.toBe(gray(b)[1]);
    }
  });

  /** True if (z, w) are a transition pair (Lee 1, same ring-class bit). */
  function isTransitionPartner(z: number, w: number): boolean {
    return w !== z && leeDistance(z, w) === 1 && gray(z)[0] === gray(w)[0];
  }

  /** True if (z, w) are a type-1 transversion pair (Lee 1, different ring-class bit). */
  function isType1TransversionPartner(z: number, w: number): boolean {
    return w !== z && leeDistance(z, w) === 1 && gray(z)[0] !== gray(w)[0];
  }

  it('each Z₄ symbol has exactly one transition partner', () => {
    for (let z = 0; z < 4; z++) {
      const transitionPartners = [0, 1, 2, 3].filter((w) => isTransitionPartner(z, w));
      expect(transitionPartners).toHaveLength(1);
    }
  });

  it('each Z₄ symbol has exactly one type-1 transversion partner', () => {
    for (let z = 0; z < 4; z++) {
      const tv1Partners = [0, 1, 2, 3].filter((w) => isType1TransversionPartner(z, w));
      expect(tv1Partners).toHaveLength(1);
    }
  });

  it('each Z₄ symbol has exactly one complement-transversion partner (its Watson–Crick complement)', () => {
    for (let z = 0; z < 4; z++) {
      const tv2Partners = [0, 1, 2, 3].filter((w) =>
        w !== z && leeDistance(z, w) === 2,
      );
      expect(tv2Partners).toHaveLength(1);
      // The partner must be the complement
      expect(tv2Partners[0]).toBe(complement(z));
    }
  });

  it('all 12 ordered Z₄ pairs are exactly partitioned into transitions, tv1, and tv2', () => {
    let tsCount = 0;
    let tv1Count = 0;
    let tv2Count = 0;
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        if (a === b) continue;
        if (leeDistance(a, b) === 2) {
          tv2Count++;
        } else if (isTransitionPartner(a, b)) {
          tsCount++;
        } else if (isType1TransversionPartner(a, b)) {
          tv1Count++;
        }
      }
    }
    // 4 transitions (G→A, A→G, C→T, T→C), 4 tv1, 4 tv2 = 12 total
    expect(tsCount).toBe(4);
    expect(tv1Count).toBe(4);
    expect(tv2Count).toBe(4);
    expect(tsCount + tv1Count + tv2Count).toBe(12);
  });
});

// ─── T0-P9: Z₄ Gray-code isometry uniqueness (alphabet optimality) ───────────
//
// The Z₄ Gray map φ: Z₄ → {0,1}² is a Lee-to-Hamming isometry:
// d_L(a,b) = d_H(φ(a), φ(b)) for all a,b ∈ Z₄.
//
// No analogous exact isometry exists for Z₈ (3 bits) because max(d_L) = 4 on
// Z₈ but max(d_H) = 3 on 3-bit strings.  Likewise for larger alphabets.
// This confirms the algebraic claim in PREDICTIONS.md §P9.

describe('T0-P9: Z₄ Gray-code isometry is exact; Z₈ and larger alphabets lack exact Lee-to-Hamming isometry', () => {
  /** Hamming distance between two non-negative integers viewed as k-bit strings. */
  function hammingDist(a: number, b: number): number {
    let x = a ^ b;
    let count = 0;
    while (x) { count += x & 1; x >>>= 1; }
    return count;
  }

  /** Gray encoding for an n-element ring (standard reflection code). */
  function reflectionGray(z: number): number {
    return z ^ (z >> 1);
  }

  it('Z₄ Gray map is a perfect Lee-to-Hamming isometry', () => {
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        const leeDist = leeDistance(a, b);
        const hammDist = hammingDist(grayEncode(a), grayEncode(b));
        expect(hammDist).toBe(leeDist);
      }
    }
  });

  it('Z₈ reflection Gray code is NOT an exact Lee-to-Hamming isometry (max Lee = 4 > max Hamming on 3 bits = 3)', () => {
    // On Z₈ the maximum Lee distance is 4 (between symbols 0 and 4).
    // A 3-bit string has maximum Hamming distance 3.
    // Therefore no bijection Z₈ → {0,1}³ can preserve Lee distance exactly.
    const maxLeeDist = Math.max(
      ...Array.from({ length: 8 }, (_, a) =>
        Math.max(...Array.from({ length: 8 }, (_, b) => {
          const diff = Math.abs(a - b);
          return Math.min(diff, 8 - diff); // Lee distance on Z₈
        })),
      ),
    );
    expect(maxLeeDist).toBe(4); // Z₈ max Lee distance

    const maxHammDist3Bits = 3; // max Hamming on 3-bit strings
    expect(maxLeeDist).toBeGreaterThan(maxHammDist3Bits);
  });

  it('Z₈ reflection Gray code underestimates Lee distance for some pairs', () => {
    // Specifically: symbols 0 and 4 have Lee distance 4 on Z₈,
    // but their Gray codes (000 and 110) have Hamming distance 2.
    const leeOn8 = (a: number, b: number): number => {
      const diff = Math.abs(a - b);
      return Math.min(diff, 8 - diff);
    };
    const lee04 = leeOn8(0, 4);
    const hamm04 = hammingDist(reflectionGray(0), reflectionGray(4));
    expect(lee04).toBe(4);
    expect(hamm04).toBeLessThan(lee04); // Gray Hamming underestimates Lee for Z₈
  });

  it('Z₄ isometry is tight: grayDecode(grayEncode(z)) = z for all z', () => {
    for (let z = 0; z < 4; z++) {
      expect(grayDecode(grayEncode(z))).toBe(z);
    }
  });

  it('Z₄ Gray codes are all distinct (bijection)', () => {
    const codes = [0, 1, 2, 3].map((z) => grayEncode(z));
    expect(new Set(codes).size).toBe(4);
  });
});
