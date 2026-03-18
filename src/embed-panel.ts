/**
 * embed-panel.ts — Embedding panel rendering
 *
 * Pure functions for rendering the Q² embedding heat-map and quantisation
 * result into the embedding panel DOM elements.
 *
 * All functions take explicit DOM element references so they are testable in
 * isolation without a global document.
 */

/** Return the minimum value in a Float32Array. */
export function min(arr: Float32Array): number {
  let m = Infinity;
  for (const v of arr) if (v < m) m = v;
  return m;
}

/** Return the maximum value in a Float32Array. */
export function max(arr: Float32Array): number {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  return m;
}

/**
 * Renders a tiny heat-map of the last-LIV-layer embeddings onto `canvas`.
 *
 * One column per sequence position, one row per hidden-dimension bin.
 * Colour: blue (negative) → white (zero) → red (positive).
 */
export function renderEmbeddingHeatmap(
  data: Float32Array,
  seqLen: number,
  hiddenDim: number,
  canvas: HTMLCanvasElement,
): void {
  const W = Math.min(seqLen, canvas.clientWidth || 320);
  const H = 64; // fixed display height

  canvas.width = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const rowsPerCell = Math.ceil(hiddenDim / H);
  const colsPerCell = Math.ceil(seqLen / W);

  const minVal = min(data);
  const maxVal = max(data);
  const range = maxVal - minVal || 1;

  for (let row = 0; row < H; row++) {
    for (let col = 0; col < W; col++) {
      // Average values in the bin.
      let sum = 0;
      let count = 0;
      for (let d = row * rowsPerCell; d < Math.min((row + 1) * rowsPerCell, hiddenDim); d++) {
        for (let s = col * colsPerCell; s < Math.min((col + 1) * colsPerCell, seqLen); s++) {
          sum += data[s * hiddenDim + d]!;
          count++;
        }
      }
      const v = count ? sum / count : 0;
      const t = (v - minVal) / range; // 0..1

      // Blue → White → Red colour map.
      const r = t > 0.5 ? 255 : Math.round(t * 2 * 255);
      const g = t > 0.5 ? Math.round((1 - t) * 2 * 255) : Math.round(t * 2 * 255);
      const b = t < 0.5 ? 255 : Math.round((1 - t) * 2 * 255);

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(col, row, 1, 1);
    }
  }
}

/**
 * Append the Q² quantisation result to `statsEl.textContent`.
 *
 * @param packed  - n/4 packed Gray-encoded bytes from q2_quantise / q2EncodeDirect
 * @param key     - 64-bit MSB-aligned transition key from q2_key / q2KeyDirect
 * @param n       - original embedding dimension
 * @param statsEl - paragraph element to update
 */
export function renderQ2Result(
  packed: Uint8Array,
  key: bigint,
  n: number,
  statsEl: HTMLParagraphElement,
): void {
  // Hex dump of the first 8 bytes (32 symbols) for display.
  const hexBytes = Array.from(packed.slice(0, 8))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  const ellipsis = packed.length > 8 ? '…' : '';
  // Display the key as a zero-padded 16-hex-digit unsigned 64-bit value.
  const keyHex = key.toString(16).padStart(16, '0');
  statsEl.textContent +=
    `\nQ²: [${hexBytes}${ellipsis}] (${n >> 2} bytes, ${n} dims)  key=0x${keyHex}`;
}
