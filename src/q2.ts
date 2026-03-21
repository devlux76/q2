/**
 * q2.ts — Q² WebAssembly Kernel Wrapper
 *
 * Loads and instantiates the Q² WASM module compiled from src/q2.wat.
 * Provides the quaternary semantic quantisation algorithm described in
 * DESIGN.md §1.5 – §2.2 for any model dimension n (a power of 2).
 *
 * Build note: `bun run build:wat` recompiles src/q2.wat → src/q2.wasm and
 * updates the WASM_B64 constant below via `scripts/embed-wat.mjs`.
 */

// ─── Dtype constants ─────────────────────────────────────────────────────────

/**
 * Numeric dtype identifiers for the q2_quantise kernel.
 *
 * The value matches the $dtype parameter expected by the WAT module.
 *
 * | ID | Name | Width      | Description                                         |
 * |----|------|-----------|-----------------------------------------------------|
 * |  0 | fp32 | 4 B/elem  | IEEE 754 single-precision float                    |
 * |  1 | fp16 | 2 B/elem  | IEEE 754 half-precision; rebiased by kernel         |
 * |  2 | q8   | 1 B/elem  | Signed int8 ∈ [−128, 127]; scale removed by L2-norm |
 * |  3 | q4   | ½ B/elem  | Packed nibbles, unsigned ∈ [0,15], centred at 8     |
 * |  4 | q2   | ¼ B/elem  | Packed 2-bit Z₄ symbols from a prior Q² pass       |
 *
 * ### Sub-fp32 bit-twiddling
 *
 * **fp16 (dtype = 1)**
 * Each element occupies 2 bytes in the buffer. The kernel uses the standard
 * IEEE 754 half-to-single-precision algorithm: sign bit is preserved; the
 * 5-bit biased exponent (bias 15) is rebiased to 127 to match fp32; the
 * 10-bit mantissa is shifted left by 13 to fill the 23-bit fp32 mantissa
 * field. Denormals (exponent = 0) are approximated as zero — their magnitude
 * is below quantisation resolution after L2 normalisation. Infinities and
 * NaNs are propagated unchanged.
 *
 * **q8 (dtype = 2)**
 * Each element is a signed 8-bit integer loaded with `i32.load8_s`.
 * The raw byte value ∈ [−128, 127] is cast directly to f32. Because the
 * entire vector is L2-normalised before thresholding, the implicit ×128
 * scale factor cancels out and does not affect the quaternary symbols
 * produced.
 *
 * **q4 (dtype = 3)**
 * Elements are packed two-per-byte as unsigned nibbles. For element index d:
 *   byte address = bufferBase + (d >> 1)
 *   value = d is even ? byte >> 4        (high nibble)
 *                     : byte & 0x0F      (low nibble)
 * The nibble ∈ [0, 15] is centred by subtracting 8, giving a signed value
 * ∈ [−8, 7]. L2 normalisation then removes the ×8 scale, leaving a
 * unit-norm vector ready for thresholding.
 *
 * **q2 (dtype = 4)**
 * The input buffer must already contain n/4 bytes of packed q2 data from a
 * prior quantisation step: each byte holds four 2-bit Gray-encoded Z₄
 * symbols (MSB-first).
 * For this dtype the kernel performs a pure byte-for-byte pass-through:
 * it copies the n/4 input bytes directly to the output and returns without
 * running any preprocessing, L2 normalisation, thresholding, or re-encoding.
 */
export type Q2Dtype = 0 | 1 | 2 | 3 | 4;

export const Q2_DTYPE_FP32: Q2Dtype = 0;
export const Q2_DTYPE_FP16: Q2Dtype = 1;
export const Q2_DTYPE_Q8:   Q2Dtype = 2;
export const Q2_DTYPE_Q4:   Q2Dtype = 3;
export const Q2_DTYPE_Q2:   Q2Dtype = 4;

/** Maps the dtype string from EmbeddingMsg / AppSettings to a Q2Dtype id. */
export const DTYPE_TO_Q2: Record<string, Q2Dtype> = {
  fp32: Q2_DTYPE_FP32,
  fp16: Q2_DTYPE_FP16,
  q8:   Q2_DTYPE_Q8,
  q4:   Q2_DTYPE_Q4,
  q2:   Q2_DTYPE_Q2,
};

// ─── WASM bytes (compiled from src/q2.wat) ───────────────────────────────────

/**
 * Base64-encoded WASM binary compiled from src/q2.wat.
 * Regenerate with: bun run build:wat
 */
const WASM_B64 =
  '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';

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64.replace(/\s+/g, ''));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ─── Kernel interface ────────────────────────────────────────────────────────

/**
 * Fixed byte offset in WASM linear memory where the host should write the
 * embedding tensor before calling {@link Q2Kernel.quantise}.
 * (Page 4, offset 0x40000 = 262144 bytes — above internal kernel buffers.)
 */
export const Q2_INPUT_OFFSET = 0x40000;

/**
 * Fixed byte offset in WASM linear memory where q2_quantise writes its
 * packed Gray-encoded output bytes (page 1, offset 0x10000 = 65536 bytes).
 */
export const Q2_OUTPUT_OFFSET = 0x10000;

/** Live Q² kernel backed by the compiled WASM module. */
export interface Q2Kernel {
  /** Shared linear memory; used to write input tensors and read results. */
  readonly memory: WebAssembly.Memory;

  /**
   * Quantise a [seq_len × n] activation tensor to n/4 packed Gray-encoded bytes,
   * using the hidden-state activation at the last token position (seq_len − 1).
   *
   * @param inputOffset - byte offset in WASM memory where the tensor is stored
   * @param seqLen      - number of token positions; the last one (seq_len − 1) is used
   * @param n           - native embedding dimension (columns); power of 2, ≤ 16 384
   * @param dtype       - element dtype (Q2_DTYPE_*)
   * @param outOffset   - byte offset in WASM memory for the output
   * @returns           number of bytes written (always n/4)
   */
  quantise(
    _inputOffset: number,
    _seqLen: number,
    _n: number,
    _dtype: Q2Dtype,
    _outOffset: number,
  ): number;

  /**
   * Derive the 64-bit MSB-aligned transition key from packed Q² bytes.
   *
   * @param packedOffset - byte offset in WASM memory for the packed bytes
   * @param n            - original dimension count
   * @returns            64-bit key as BigInt (DESIGN.md §2.2)
   */
  key(_packedOffset: number, _n: number): bigint;

  /**
   * Compute the Lee distance between two packed Gray-encoded Q² vectors.
   *
   * Uses SIMD-accelerated XOR + popcnt to exploit the Gray-map isometry
   * (DESIGN.md §2.7 Theorem 2.1): d_H(φ(u), φ(v)) = d_L(u, v).
   *
   * @param aOffset - byte offset in WASM memory for the first packed vector
   * @param bOffset - byte offset in WASM memory for the second packed vector
   * @param n       - original embedding dimension (n/4 packed bytes per vector)
   * @returns       total Lee distance (sum of per-dimension distances)
   */
  leeDistance(_aOffset: number, _bOffset: number, _n: number): number;
}

// ─── Instantiation ────────────────────────────────────────────────────────────

let kernelPromise: Promise<Q2Kernel> | null = null;

/**
 * Returns a singleton Q2Kernel instance, instantiating the WASM module on the
 * first call.  Subsequent calls return the same promise.
 */
export function getKernel(): Promise<Q2Kernel> {
  kernelPromise ??= instantiate();
  return kernelPromise;
}

async function instantiate(): Promise<Q2Kernel> {
  const bytes = b64ToBytes(WASM_B64);
  const { instance } = await WebAssembly.instantiate(
    bytes.buffer as ArrayBuffer,
    {},
  );

  type WasmExports = {
    mem: WebAssembly.Memory;
    q2_quantise: (_ip: number, _sl: number, _n: number, _dt: number, _op: number) => number;
    q2_key: (_sp: number, _n: number) => bigint;
    q2_lee_distance: (_ap: number, _bp: number, _n: number) => number;
  };
  const e = instance.exports as WasmExports;

  return {
    memory: e.mem,
    quantise(inputOffset, seqLen, n, dtype, outOffset) {
      return e.q2_quantise(inputOffset, seqLen, n, dtype, outOffset);
    },
    key(packedOffset, n) {
      return e.q2_key(packedOffset, n);
    },
    leeDistance(aOffset, bOffset, n) {
      return e.q2_lee_distance(aOffset, bOffset, n);
    },
  };
}

// ─── Pure-TypeScript fallback ─────────────────────────────────────────────────

/**
 * Pure-TypeScript reference implementation of the Q² kernel.
 *
 * Used by unit tests (no WASM runtime required) and as a verified reference
 * against which the WAT implementation is validated.
 */
export interface Q2Result {
  /** n/4 packed Gray-encoded bytes. */
  packed: Uint8Array;
  /** 64-bit MSB-aligned transition key. */
  key: bigint;
}

/**
 * Quantise an L2-normalised Float32Array embedding vector of shape [n]
 * using the Q² algorithm and return the packed bytes and transition key.
 *
 * This function operates entirely in TypeScript and does not require the WASM
 * module; it is suitable for unit tests and server-side usage.
 *
 * @param vec    - L2-normalised float32 vector of length n (power of 2)
 * @param n      - dimension count (must equal vec.length)
 * @returns      packed bytes and 64-bit key
 */
export function q2EncodeDirect(vec: Float32Array, n: number): Q2Result {
  // Threshold: Φ⁻¹(3/4) / √n ≈ 0.6745 / √n  (DESIGN.md §1.5)
  const tau = 0.6745 / Math.sqrt(n);
  const nBytes = n >> 2;
  const packed = new Uint8Array(nBytes);

  for (let d = 0; d < n; d++) {
    const v = vec[d] ?? 0;
    let sym: number;
    if (v <= -tau)       sym = 0; // A: strong negative
    else if (v <= 0)     sym = 1; // B: weak negative
    else if (v <= tau)   sym = 2; // C: weak positive
    else                 sym = 3; // D: strong positive

    // Gray-encode: g = sym ⊕ (sym >> 1)  (DESIGN.md §1.7)
    const g = sym ^ (sym >> 1);

    // Pack MSB-first within each byte: out[d/4] |= g << (2 * (3 - d%4))
    const byteIdx = d >> 2;
    const shift = (3 - (d & 3)) << 1;
    packed[byteIdx]! |= g << shift;
  }

  return { packed, key: q2KeyDirect(packed, n) };
}

/**
 * Derive the 64-bit transition key from n/4 packed Gray-encoded bytes.
 *
 * @param packed - n/4 packed Gray-encoded bytes (output of q2EncodeDirect)
 * @param n      - original dimension count
 * @returns      64-bit MSB-aligned key as BigInt
 */
export function q2KeyDirect(packed: Uint8Array, n: number): bigint {
  let key = 0n;
  let trans = 0;
  let prev = 0xFF; // sentinel: no previous symbol

  for (let d = 0; d < n; d++) {
    const byteIdx = d >> 2;
    const shift = (3 - (d & 3)) << 1;
    const g = ((packed[byteIdx] ?? 0) >> shift) & 0x3;

    // Decode Gray → Z₄: z = (g & 2) | ((g >> 1) ⊕ (g & 1))
    const z = (g & 2) | ((g >> 1) ^ (g & 1));

    if (z !== prev) {
      prev = z;
      if (trans < 32) {
        key |= BigInt(z) << BigInt(62 - 2 * trans);
        trans++;
      }
    }
  }

  return key;
}

/**
 * L2-normalise a Float32Array activation vector of logical shape [n].
 *
 * The first `n` elements of `data` are copied into a new Float32Array; if
 * `data.length < n`, the remaining elements are treated as 0 and padded with
 * zeros. Any elements beyond index `n - 1` in `data` are ignored.
 *
 * @param data - raw activation values at the selected token position
 * @param n    - embedding dimension (length of the returned vector)
 * @returns    L2-normalised Float32Array of length n
 */
export function l2Normalise(data: Float32Array, n: number): Float32Array {
  const v = new Float32Array(n);
  for (let d = 0; d < n; d++) v[d] = data[d] ?? 0;

  let normSq = 0;
  for (let d = 0; d < n; d++) { const x = v[d]!; normSq += x * x; }
  if (normSq > 1e-16) {
    const normInv = 1 / Math.sqrt(normSq);
    for (let d = 0; d < n; d++) v[d]! *= normInv;
  }

  return v;
}

/**
 * Compute the Lee distance between two packed Gray-encoded Q² vectors.
 *
 * This is a convenience re-export of {@link leeDistancePacked} from q2stats.ts.
 * Both exploit the Gray-map isometry (DESIGN.md §2.7 Theorem 2.1):
 *   d_H(φ(u), φ(v)) = d_L(u, v)
 *
 * Hamming distance on Gray-encoded bits equals Lee distance on Z₄ originals,
 * so the total distance is simply popcount(a XOR b) over all packed bytes.
 * Distance is computed over min(a.length, b.length) bytes.
 *
 * @param a - first  packed Gray-encoded vector (n/4 bytes)
 * @param b - second packed Gray-encoded vector (n/4 bytes)
 * @returns total Lee distance (sum of per-dimension Lee distances)
 */
export { leeDistancePacked as q2LeeDistanceDirect } from './q2stats.js';
