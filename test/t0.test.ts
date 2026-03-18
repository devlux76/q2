/**
 * t0.test.ts — T0 Phase: Unit Tests and Invariants
 *
 * Tests the algebraic invariants of the Q² encoding that must hold before any
 * corpus or model experiment.  These tests run entirely in TypeScript, require
 * no model download, and should pass on any machine.
 *
 * References: TESTING.md §T0, PREDICTIONS.md §P1, §P2, §P3, §P8, §P10.
 *
 * Test organisation:
 *   T0-P1  — CGAT mapping and complement involution
 *   T0-Q   — Quantisation and packing invariants (smoke tests)
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
  it('nullCollisionExpectation for 1000 keys in 64-bit space is < 0.001', () => {
    // 1000*(999) / 2^65 ≈ 2.7e-11 — essentially zero
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
