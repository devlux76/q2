import { describe, expect, it } from 'vitest';
import {
  q2EncodeDirect,
  q2KeyDirect,
  q2LeeDistanceDirect,
  l2Normalise,
  Q2_DTYPE_FP32,
  Q2_DTYPE_FP16,
  Q2_DTYPE_Q8,
  Q2_DTYPE_Q4,
  Q2_DTYPE_Q2,
  DTYPE_TO_Q2,
} from '../src/q2.ts';

// ─── Gray-encoding table (DESIGN.md §1.7) ────────────────────────────────────
// Symbol → Z₄ → Gray: A=0→00, B=1→01, C=2→11, D=3→10

describe('q2EncodeDirect', () => {
  it('produces n/4 packed bytes for n=8', () => {
    const vec = new Float32Array(8).fill(0); // all zero → B (weak negative, v ≤ 0)
    const { packed } = q2EncodeDirect(vec, 8);
    expect(packed.length).toBe(2);
  });

  it('encodes strong-positive (D) as Gray 10₂', () => {
    // All dimensions strongly positive (above threshold) in a pre-normalised vector.
    // After L2-normalisation each element becomes 1/√8 ≈ 0.354.
    // τ* = 0.6745/√8 ≈ 0.238, so 0.354 > τ* → all D.
    const vec = new Float32Array(8).fill(1 / Math.sqrt(8));
    const { packed } = q2EncodeDirect(vec, 8);
    // D → 3 → Gray 10₂ = 0b10 = 2
    // Each byte: 4 × 10₂ = 10101010₂ = 0xAA
    expect(packed[0]).toBe(0xAA);
    expect(packed[1]).toBe(0xAA);
  });

  it('encodes strong-negative (A) as Gray 00₂', () => {
    const vec = new Float32Array(8).fill(-1 / Math.sqrt(8));
    const { packed } = q2EncodeDirect(vec, 8);
    // A → 0 → Gray 00₂ = 0
    // Each byte: 4 × 00₂ = 00000000₂ = 0x00
    expect(packed[0]).toBe(0x00);
    expect(packed[1]).toBe(0x00);
  });

  it('encodes mixed ABCD correctly for known input', () => {
    // Pre-normalised vector with known quartile positions
    // tau* = 0.6745 / sqrt(8) ≈ 0.2384
    // Assign each dimension to a known symbol
    const tau = 0.6745 / Math.sqrt(8);
    // d0: > tau → D (3 → 10₂)
    // d1: < -tau → A (0 → 00₂)
    // d2: in (0, tau] → C (2 → 11₂)
    // d3: in (-tau, 0] → B (1 → 01₂)
    // d4-d7: repeat the same pattern
    const vec = new Float32Array([
      tau + 0.1, -(tau + 0.1), tau * 0.5, -(tau * 0.5),
      tau + 0.1, -(tau + 0.1), tau * 0.5, -(tau * 0.5),
    ]);
    // L2 normalise manually (q2EncodeDirect takes a pre-normalised vec)
    let normSq = 0;
    for (const v of vec) normSq += v * v;
    const normInv = 1 / Math.sqrt(normSq);
    const normed = vec.map((v) => v * normInv);
    // Re-check symbols after normalisation
    const tauN = 0.6745 / Math.sqrt(8);
    const syms = normed.map((v) => {
      if (v <= -tauN) return 0; // A
      if (v <= 0) return 1;     // B
      if (v <= tauN) return 2;  // C
      return 3;                 // D
    });
    const grays = syms.map((s) => s ^ (s >> 1));

    const { packed } = q2EncodeDirect(new Float32Array(normed), 8);

    // Verify each byte by re-packing manually
    const expectedByte0 = (grays[0]! << 6) | (grays[1]! << 4) | (grays[2]! << 2) | grays[3]!;
    const expectedByte1 = (grays[4]! << 6) | (grays[5]! << 4) | (grays[6]! << 2) | grays[7]!;
    expect(packed[0]).toBe(expectedByte0);
    expect(packed[1]).toBe(expectedByte1);
  });

  it('key holds the single symbol in MSBs for a constant vector', () => {
    const vec = new Float32Array(8).fill(1 / Math.sqrt(8));
    const { key } = q2EncodeDirect(vec, 8);
    // All D (3 → 10₂); one distinct symbol emitted at the first position.
    // run-reduction emits r[0] = 3; key = 3n << 62n (symbol in bits 63:62).
    expect(key).toBe(BigInt.asUintN(64, 3n << 62n));
  });
});

describe('q2KeyDirect', () => {
  it('produces 0 key for empty packed bytes (n=0)', () => {
    expect(q2KeyDirect(new Uint8Array(0), 0)).toBe(0n);
  });

  it('MSB-aligns first transition in bits 63:62', () => {
    // Single-symbol vector: all D (Gray 10₂ = 0b10)
    // Byte: 4 × 10₂ = 10101010₂ = 0xAA
    const packed = new Uint8Array([0xAA, 0xAA]);
    const key = q2KeyDirect(packed, 8);
    // One transition: z=3, key = 3n << 62n
    expect(key).toBe(BigInt.asUintN(64, 3n << 62n));
  });

  it('handles maximum 32 transitions', () => {
    // Alternating A (00) and D (10): 10 00 10 00 ...
    // In bits: each pair is 10 or 00; pack 4 per byte
    // Pair pattern: D A D A D A D A → Gray: 10 00 10 00 10 00 10 00
    // Byte: 10_00_10_00 = 0b10001000 = 0x88
    const packed = new Uint8Array(8).fill(0x88); // 32 symbols
    const key = q2KeyDirect(packed, 32);
    // Transitions at every symbol (alternating D=3 and A=0)
    // r[0]=3, r[1]=0, r[2]=3, r[3]=0, ...  (16 pairs × 2 = 32 transitions, but capped at 32)
    // key bits (MSB-aligned): 10 00 10 00 ... (16 pairs of 10+00)
    // r[0]=3=10₂ → bits 63:62
    // r[1]=0=00₂ → bits 61:60
    // r[2]=3=10₂ → bits 59:58
    // etc.
    expect(key & (3n << 62n)).toBe(3n << 62n); // first transition is D (3)
    expect(key & (3n << 60n)).toBe(0n);         // second transition is A (0)
  });

  it('Gray-decode round-trip: encode then key gives consistent symbol in key MSBs', () => {
    // Use q2EncodeDirect to produce packed bytes and then call q2KeyDirect
    // independently on the packed bytes — should match the key from q2EncodeDirect.
    const vec = new Float32Array(16).map((_, i) => (i % 4 < 2 ? 0.5 : -0.5));
    const normSq = vec.reduce((s, v) => s + v * v, 0);
    const normInv = 1 / Math.sqrt(normSq);
    const normed = new Float32Array(vec.map((v) => v * normInv));

    const { packed, key: keyFromEncode } = q2EncodeDirect(normed, 16);
    const keyFromDirect = q2KeyDirect(packed, 16);

    expect(keyFromDirect).toBe(keyFromEncode);
  });
});

describe('l2Normalise', () => {
  it('normalises a unit vector along dim 0 to itself', () => {
    const n = 4;
    const data = new Float32Array([1, 0, 0, 0]);
    const v = l2Normalise(data, n);
    expect(v.length).toBe(n);
    expect(v[0]).toBeCloseTo(1, 5);
    for (let i = 1; i < n; i++) expect(v[i]).toBeCloseTo(0, 5);
  });

  it('normalises a scaled vector to unit length', () => {
    const n = 4;
    const data = new Float32Array([3, 4, 0, 0]); // norm = 5
    const v = l2Normalise(data, n);
    expect(v[0]).toBeCloseTo(0.6, 5);
    expect(v[1]).toBeCloseTo(0.8, 5);
    expect(v[2]).toBeCloseTo(0, 5);
    expect(v[3]).toBeCloseTo(0, 5);
  });

  it('returns unit vector (L2 norm ≈ 1)', () => {
    const n = 8;
    const data = new Float32Array(n).map((_, i) => Math.sin(i + 1));
    const v = l2Normalise(data, n);
    const normSq = Array.from(v).reduce((s, x) => s + x * x, 0);
    expect(normSq).toBeCloseTo(1, 5);
  });

  it('handles zero vector gracefully (no NaN)', () => {
    const n = 4;
    const data = new Float32Array(n); // all zeros
    const v = l2Normalise(data, n);
    for (const x of v) expect(isFinite(x)).toBe(true);
  });
});

describe('q2LeeDistanceDirect', () => {
  it('returns 0 for identical vectors', () => {
    // All D: Gray 10₂ → 0xAA per byte
    const a = new Uint8Array([0xAA, 0xAA]);
    expect(q2LeeDistanceDirect(a, a, 8)).toBe(0);
  });

  it('returns maximum distance (2) for complement pairs A↔C', () => {
    // A=00₂, C=11₂ → complement (DESIGN.md §2.8, distance 2 per dim)
    // All A → 0x00; All C → 0xFF
    const a = new Uint8Array([0x00, 0x00]); // 8× A
    const c = new Uint8Array([0xFF, 0xFF]); // 8× C (11₂ packed)
    // Each of 8 dims differs by 2 bits → total Hamming = 16 = total Lee
    expect(q2LeeDistanceDirect(a, c, 8)).toBe(16);
  });

  it('returns 1 per dimension for adjacent symbols A↔B', () => {
    // A=00₂, B=01₂ → adjacent, Lee distance 1 per dim
    // All A → 0x00; All B → 01_01_01_01₂ = 0x55
    const a = new Uint8Array([0x00, 0x00]); // 8× A
    const b = new Uint8Array([0x55, 0x55]); // 8× B
    expect(q2LeeDistanceDirect(a, b, 8)).toBe(8);
  });

  it('returns 1 per dimension for cyclic-adjacent D↔A (strong extremes)', () => {
    // D=10₂, A=00₂ → cyclic adjacent, Lee distance 1 per dim
    // XOR = 10₂ per dim → 1 bit per dim → Hamming 1 = Lee 1
    const d = new Uint8Array([0xAA, 0xAA]); // 8× D (10₂)
    const a = new Uint8Array([0x00, 0x00]); // 8× A (00₂)
    expect(q2LeeDistanceDirect(d, a, 8)).toBe(8);
  });

  it('is symmetric: distance(a,b) = distance(b,a)', () => {
    const a = new Uint8Array([0xAA, 0x55]); // mixed D and B
    const b = new Uint8Array([0xFF, 0x00]); // mixed C and A
    expect(q2LeeDistanceDirect(a, b, 8)).toBe(q2LeeDistanceDirect(b, a, 8));
  });

  it('works with n=0 (empty vectors)', () => {
    expect(q2LeeDistanceDirect(new Uint8Array(0), new Uint8Array(0), 0)).toBe(0);
  });
});

describe('Q2Dtype constants and DTYPE_TO_Q2 map', () => {
  it('maps dtype strings to correct ids', () => {
    expect(DTYPE_TO_Q2['fp32']).toBe(Q2_DTYPE_FP32);
    expect(DTYPE_TO_Q2['fp16']).toBe(Q2_DTYPE_FP16);
    expect(DTYPE_TO_Q2['q8']).toBe(Q2_DTYPE_Q8);
    expect(DTYPE_TO_Q2['q4']).toBe(Q2_DTYPE_Q4);
    expect(DTYPE_TO_Q2['q2']).toBe(Q2_DTYPE_Q2);
  });

  it('dtype ids are 0..4', () => {
    expect(Q2_DTYPE_FP32).toBe(0);
    expect(Q2_DTYPE_FP16).toBe(1);
    expect(Q2_DTYPE_Q8).toBe(2);
    expect(Q2_DTYPE_Q4).toBe(3);
    expect(Q2_DTYPE_Q2).toBe(4);
  });
});
