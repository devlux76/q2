/**
 * q2stats.ts — Q² Statistical Analysis Functions
 *
 * Pure-TypeScript analysis functions for testing the hypotheses in PREDICTIONS.md.
 * These functions operate on transition sequences derived from the Q² kernel output
 * and do not require a WASM runtime or model download.
 *
 * ## Relevant predictions
 * - P1: CGAT complement involution and Gray-code isometry
 * - P2: Hairpin density ρ_hp
 * - P3: Complement bigram suppression
 * - P5: Reverse-complement antonym retrieval
 * - P8: Codon (triplet) frequency distribution
 * - P10: 64-bit key collision rate
 *
 * All T0 unit tests and T1 null-distribution baselines use only functions from
 * this module and from q2.ts — no external model or corpus is required.
 */

// ─── P1: Complement involution and Lee metric ────────────────────────────────

/**
 * The complement involution θ on Z₄ (DESIGN.md §2.8, PREDICTIONS.md §P1).
 *
 * Watson–Crick complementarity: G↔C (0↔2) and A↔T (1↔3).
 * Algebraically: θ(z) = z ⊕ 2 (flip the purine/pyrimidine bit).
 *
 * @param z - Z₄ symbol in {0, 1, 2, 3}
 * @returns complement symbol
 */
export function complement(z: number): number {
  return (z ^ 2) & 3;
}

/**
 * Lee distance on Z₄ (DESIGN.md §1.3).
 *
 * d_L(a, b) = min(|a − b| mod 4, 4 − |a − b| mod 4)
 *
 * Values:
 * - 0: same symbol
 * - 1: adjacent on Z₄ cycle (G–A, A–T, T–C, C–G on the ring)
 * - 2: opposite (complement pair, G–C or A–T)
 *
 * @param a - Z₄ symbol in {0, 1, 2, 3}
 * @param b - Z₄ symbol in {0, 1, 2, 3}
 * @returns Lee distance in {0, 1, 2}
 */
export function leeDistance(a: number, b: number): number {
  const diff = Math.abs((a & 3) - (b & 3));
  return Math.min(diff, 4 - diff);
}

/**
 * Lee distance between two packed Q² byte arrays.
 *
 * Because the Gray map is a Lee-to-Hamming isometry on Z₄ (DESIGN.md §2.7),
 * the Lee distance over the full n-dimensional symbol vector equals the
 * Hamming distance over the packed Gray-encoded bytes, which equals
 * `popcount(a XOR b)`.
 *
 * This is the inner-product kernel used by the two-stage retrieval architecture
 * (§P-6) and can be implemented with SIMD popcnt instructions.
 *
 * The distance is computed over the first `min(a.length, b.length)` bytes.
 * Any extra bytes in the longer array are ignored; in most callers this should
 * only happen if there is a bug or mismatch in the packed vector dimensions.
 *
 * @param a - packed Q² bytes (typically n/4 bytes, output of q2EncodeDirect)
 * @param b - packed Q² bytes for a vector of (typically) the same dimension
 * @returns total Lee distance (sum over all compared symbol positions)
 */
export function leeDistancePacked(a: Uint8Array, b: Uint8Array): number {
  const len = Math.min(a.length, b.length);
  let dist = 0;
  for (let i = 0; i < len; i++) {
    let v = (a[i]! ^ b[i]!) & 0xFF;
    // Kernighan's bit-count
    while (v !== 0) { v &= v - 1; dist++; }
  }
  return dist;
}

// ─── P1: Gray-code isometry check ────────────────────────────────────────────

/**
 * Gray-encode a Z₄ symbol to its 2-bit Gray code (DESIGN.md §1.7).
 * g = z ⊕ (z >> 1)
 */
export function grayEncode(z: number): number {
  return (z ^ (z >> 1)) & 3;
}

/**
 * Gray-decode a 2-bit Gray code to a Z₄ symbol.
 * z = (g & 2) | ((g >> 1) ^ (g & 1))
 */
export function grayDecode(g: number): number {
  return ((g & 2) | ((g >> 1) ^ (g & 1))) & 3;
}

// ─── Symbol unpacking ─────────────────────────────────────────────────────────

/**
 * Unpack n Gray-decoded Z₄ symbols from packed Q² bytes.
 *
 * Each byte holds 4 symbols, packed MSB-first: bits 7:6 → d=0, 5:4 → d=1,
 * 3:2 → d=2, 1:0 → d=3.  The 2-bit Gray code is decoded back to Z₄.
 *
 * @param packed - n/4 packed bytes (output of q2EncodeDirect / q2_quantise)
 * @param n      - original dimension count
 * @returns      array of n Z₄ symbols in {0, 1, 2, 3}
 */
export function unpackSymbols(packed: Uint8Array, n: number): number[] {
  const symbols: number[] = [];
  for (let d = 0; d < n; d++) {
    const byteIdx = d >> 2;
    const shift = (3 - (d & 3)) << 1;
    const g = ((packed[byteIdx] ?? 0) >> shift) & 0x3;
    symbols.push(grayDecode(g));
  }
  return symbols;
}

// ─── P2, P3, P8: Transition sequence statistics ───────────────────────────────

/**
 * Run-reduce a symbol array to the transition sequence R (DESIGN.md §3.1).
 *
 * Consecutive duplicate symbols are collapsed to a single occurrence.
 * The result satisfies r_i ≠ r_{i+1} for all i.
 *
 * @param symbols - raw Z₄ symbol array (e.g. from unpackSymbols)
 * @returns       run-reduced transition sequence
 */
export function runReduce(symbols: number[]): number[] {
  const seq: number[] = [];
  let prev = -1;
  for (const s of symbols) {
    if (s !== prev) { seq.push(s); prev = s; }
  }
  return seq;
}

/**
 * Hairpin density ρ_hp(R) — PREDICTIONS.md §P2.
 *
 * Counts triplets (r_i, r_{i+1}, r_i) where r_{i+1} = θ(r_i): the embedding
 * trajectory visits the semantic antipode and returns.
 *
 *   ρ_hp = |{i : r_{i+1} = θ(r_i) and r_{i+2} = r_i}| / (|R| − 2)
 *
 * Null expectation for a uniformly random transition sequence: 1/9 ≈ 0.111.
 *
 * @param seq - run-reduced transition sequence (r_i ≠ r_{i+1} for all i)
 * @returns   hairpin density in [0, 1], or 0 for sequences shorter than 3
 */
export function hairpinDensity(seq: number[]): number {
  if (seq.length < 3) return 0;
  let count = 0;
  for (let i = 0; i < seq.length - 2; i++) {
    const ri = seq[i]!;
    const ri1 = seq[i + 1]!;
    const ri2 = seq[i + 2]!;
    if (ri1 === complement(ri) && ri2 === ri) count++;
  }
  return count / (seq.length - 2);
}

/**
 * Complement bigram frequency — PREDICTIONS.md §P3.
 *
 * Counts consecutive pairs (r_i, r_{i+1}) where r_{i+1} = θ(r_i): the Lee-
 * distance-2 transition analogous to a CpG dinucleotide in genomic DNA.
 *
 * Null expectation for a uniformly random transition sequence: 1/3 ≈ 0.333.
 * Prediction: real-corpus frequency < 1/3.
 *
 * @param seq - run-reduced transition sequence
 * @returns   complement bigram frequency in [0, 1], or 0 for length < 2
 */
export function complementBigramFreq(seq: number[]): number {
  if (seq.length < 2) return 0;
  let count = 0;
  for (let i = 0; i < seq.length - 1; i++) {
    if (seq[i + 1] === complement(seq[i]!)) count++;
  }
  return count / (seq.length - 1);
}

/**
 * Triplet (codon) frequency distribution — PREDICTIONS.md §P8.
 *
 * Counts occurrences of each consecutive triplet (r_i, r_{i+1}, r_{i+2}) in
 * the run-reduced transition sequence.  The key is the triplet encoded as a
 * three-digit decimal string "XYZ" where X, Y, Z ∈ {0,1,2,3}.
 *
 * Null expectation: approximately uniform over the 36 valid triplets.
 * A run-reduced sequence has r_i ≠ r_{i+1} for all i, so y ≠ x and z ≠ y,
 * giving 4 × 3 × 3 = 36 possible triplet forms (z may equal x).
 *
 * @param seq - run-reduced transition sequence
 * @returns   frequency map from triplet string to count
 */
export function tripletFreqs(seq: number[]): Record<string, number> {
  const freqs: Record<string, number> = {};
  for (let i = 0; i < seq.length - 2; i++) {
    const key = `${seq[i]!}${seq[i + 1]!}${seq[i + 2]!}`;
    freqs[key] = (freqs[key] ?? 0) + 1;
  }
  return freqs;
}

/**
 * Bigram frequency distribution.
 *
 * Counts occurrences of each consecutive bigram (r_i, r_{i+1}) in the
 * run-reduced transition sequence.  Key: "XY" where X, Y ∈ {0,1,2,3}.
 *
 * @param seq - run-reduced transition sequence
 * @returns   frequency map from bigram string to count
 */
export function bigramFreqs(seq: number[]): Record<string, number> {
  const freqs: Record<string, number> = {};
  for (let i = 0; i < seq.length - 1; i++) {
    const key = `${seq[i]!}${seq[i + 1]!}`;
    freqs[key] = (freqs[key] ?? 0) + 1;
  }
  return freqs;
}

// ─── P5: Reverse complement ───────────────────────────────────────────────────

/**
 * Reverse complement of a transition sequence — PREDICTIONS.md §P5.
 *
 * Applies θ to every symbol and reverses the order, analogous to the biological
 * reverse complement of a DNA strand.  Prediction: querying a corpus with the
 * reverse-complement key of a document retrieves its semantic antonym.
 *
 * @param seq - transition sequence
 * @returns   reverse complement sequence
 */
export function reverseComplementSeq(seq: number[]): number[] {
  return seq.map((z) => complement(z)).reverse();
}

// ─── P10: Key collision statistics ───────────────────────────────────────────

/**
 * Key collision statistics — PREDICTIONS.md §P10.
 *
 * For a corpus of N 64-bit keys, counts how many are duplicates.
 * The random-hash null expectation for N keys and a 64-bit space is
 * approximately N²/2^65.
 *
 * @param keys - array of 64-bit transition keys (as BigInt)
 * @returns    { collisions: number of colliding key instances (above 1 per group),
 *               groups: number of distinct keys that appear more than once,
 *               rate: collisions / total }
 */
export function collisionStats(keys: bigint[]): {
  collisions: number;
  groups: number;
  rate: number;
} {
  if (keys.length === 0) return { collisions: 0, groups: 0, rate: 0 };

  const counts = new Map<bigint, number>();
  for (const k of keys) {
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }

  let collisions = 0;
  let groups = 0;
  for (const c of counts.values()) {
    if (c > 1) {
      collisions += c - 1;
      groups++;
    }
  }
  return { collisions, groups, rate: collisions / keys.length };
}

/**
 * Theoretical null collision rate for N keys in a 2^bits space.
 *
 * By the birthday paradox approximation: P(>=1 collision) ~= 1 - e^(-N^2/2^(bits+1))
 * Expected collision count ~= N*(N-1) / 2^(bits+1)
 *
 * @param n    - number of keys
 * @param bits - number of bits in the key (default 64)
 * @returns    expected number of collisions (pairs)
 */
export function nullCollisionExpectation(n: number, bits = 64): number {
  return (n * (n - 1)) / Math.pow(2, bits + 1);
}

// ─── P4: Bigram type classification and weighted Lee distance ─────────────────

/**
 * Classify a consecutive bigram in a transition sequence by its mutation type
 * under the CGAT mapping (PREDICTIONS.md §P4).
 *
 * Under the Gray code, bit-0 (MSB) encodes purine/pyrimidine and bit-1 (LSB)
 * encodes keto/amino:
 *
 * - **Ti (transition):**    Lee distance 1; same ring-class (purine↔purine or
 *   pyrimidine↔pyrimidine); only the keto/amino bit changes.  Pairs: G↔A, C↔T.
 * - **Tv1 (transversion type 1):** Lee distance 1; different ring-class; only the
 *   purine/pyrimidine bit changes.  Pairs: G↔T, A↔C.
 * - **Tv2 (complement transversion):** Lee distance 2; both chemical bits change.
 *   Watson–Crick complement pairs: G↔C, A↔T.
 *
 * @param a - Z₄ symbol in {0, 1, 2, 3}
 * @param b - Z₄ symbol in {0, 1, 2, 3}
 * @returns bigram mutation class, or 'same' if a === b
 */
export function bigramType(a: number, b: number): 'Ti' | 'Tv1' | 'Tv2' | 'same' {
  if (a === b) return 'same';
  if (leeDistance(a, b) === 2) return 'Tv2';
  // Lee distance 1 — distinguish by ring-class (MSB of Gray code)
  const msb = (z: number) => (grayEncode(z) >> 1) & 1;
  return msb(a) === msb(b) ? 'Ti' : 'Tv1';
}

/**
 * Weighted Lee distance between two symbol arrays — PREDICTIONS.md §P4.
 *
 * Each symbol-pair contributes a weight according to its mutation class:
 * - Ti (transition, Lee 1):         `weights.Ti`
 * - Tv1 (transversion type 1, Lee 1): `weights.Tv1`
 * - Tv2 (complement transversion, Lee 2): `weights.Tv2`
 * - Same symbol: 0
 *
 * Uniform Lee distance corresponds to `{ Ti: 1, Tv1: 1, Tv2: 2 }`.
 * The prediction in §P4 is that `{ Ti: 1, Tv1: ≈1.5, Tv2: ≈3 }` outperforms
 * uniform Lee on retrieval benchmarks.
 *
 * @param a       - first Z₄ symbol array
 * @param b       - second Z₄ symbol array (length need not equal a.length)
 * @param weights - per-class weights { Ti, Tv1, Tv2 }
 * @returns total weighted distance over the overlapping prefix
 */
export function weightedLeeDistanceSeq(
  a: number[],
  b: number[],
  weights: { Ti: number; Tv1: number; Tv2: number },
): number {
  const len = Math.min(a.length, b.length);
  let dist = 0;
  for (let i = 0; i < len; i++) {
    const type = bigramType(a[i]!, b[i]!);
    if (type !== 'same') dist += weights[type];
  }
  return dist;
}

// ─── P7: Nussinov secondary structure ────────────────────────────────────────

/**
 * Nussinov score — maximum number of non-crossing complement pairs in a
 * transition sequence (PREDICTIONS.md §P7).
 *
 * A **complement pair** at positions (i, k) with i + 2 ≤ k requires:
 * - `seq[i] === seq[k]`            (same symbol at both anchors)
 * - `seq[i+1] === θ(seq[i])`       (interior opens on complement)
 * - `seq[k-1] === θ(seq[k])`       (interior closes on complement)
 *
 * The Nussinov dynamic programming finds the maximum set of such pairs with
 * no crossing: for pairs (i,k) and (j,l), either i<j<l<k (nested) or
 * i<k<j<l (sequential) — never i<j<k<l (crossing).
 *
 * A minimum valid hairpin (x, θ(x), x) at positions (i, i+2) is the simplest
 * complement pair: it corresponds to the complement palindrome codon measured
 * by `hairpinDensity`.  Nested hairpins produce scores > 1.
 *
 * Complexity: O(n³) time, O(n²) space.  Input must be at most 2 000 symbols;
 * a RangeError is thrown for longer sequences.  Use a windowed or greedy
 * heuristic (e.g. analyse the first 2 000 symbols) for longer texts.
 *
 * @param seq - run-reduced (or raw) transition sequence (length ≤ 2 000)
 * @returns   maximum number of non-crossing complement pairs
 * @throws    {RangeError} if `seq.length > 2000`
 */
export function nussinovScore(seq: number[]): number {
  const n = seq.length;
  if (n > 2000) {
    throw new RangeError(
      `nussinovScore: sequence length ${n} exceeds the 2 000-symbol cap. ` +
      'Truncate the input or analyse a representative 2 000-symbol window.',
    );
  }
  if (n < 3) return 0;

  // Allocate dp[i][j] = max pairs in seq[i..j] (inclusive)
  const dp: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  /** True iff (i, k) is a valid complement pair. */
  function isPair(i: number, k: number): boolean {
    return (
      k - i >= 2 &&
      seq[i] === seq[k] &&
      seq[i + 1] === complement(seq[i]!) &&
      seq[k - 1] === complement(seq[k]!)
    );
  }

  // Fill by increasing subsequence length
  for (let len = 2; len < n; len++) {
    for (let i = 0; i + len < n; i++) {
      const j = i + len;
      // Option 1: position j is unpaired
      dp[i]![j] = dp[i]![j - 1]!;
      // Option 2: pair j with some k in [i, j-2]
      for (let k = i; k + 2 <= j; k++) {
        if (isPair(k, j)) {
          const left = k > i ? dp[i]![k - 1]! : 0;
          const inner = k + 1 <= j - 1 ? dp[k + 1]![j - 1]! : 0;
          const score = 1 + left + inner;
          if (score > dp[i]![j]!) dp[i]![j] = score;
        }
      }
    }
  }

  return dp[0]![n - 1]!;
}

// ─── Lee distance between transition sequences ───────────────────────────────

/**
 * Sum of per-symbol Lee distances between two symbol arrays.
 *
 * For equal-length arrays, this is the total Lee distance over all positions.
 * For unequal-length arrays, only the overlapping prefix is compared.
 *
 * @param a - first Z₄ symbol array
 * @param b - second Z₄ symbol array
 * @returns total Lee distance
 */
export function leeDistanceSeq(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let dist = 0;
  for (let i = 0; i < len; i++) {
    dist += leeDistance(a[i]!, b[i]!);
  }
  return dist;
}
