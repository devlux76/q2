// Shared message types between the main thread and the inference Web Worker.
// All communication uses structured-clone-compatible payloads.

// ─── Main thread → Worker ────────────────────────────────────────────────────

export interface LoadModelMsg {
  type: 'load';
  modelId: string;
  /** ONNX file suffix: 'q4' | 'q8' | 'fp16' | 'fp32'. Default: 'q4'. */
  dtype: string;
  /** Optional HuggingFace API token for private models and higher rate limits. */
  apiToken?: string;
}

export interface GenerateMsg {
  type: 'generate';
  messages: ChatMessage[];
  config: GenerationConfig;
}

export interface AbortMsg {
  type: 'abort';
}

export type WorkerInMsg = LoadModelMsg | GenerateMsg | AbortMsg;

// ─── Worker → Main thread ────────────────────────────────────────────────────

export interface StatusMsg {
  type: 'status';
  status: 'loading' | 'ready' | 'generating' | 'idle';
  detail?: string;
}

export interface ProgressMsg {
  type: 'progress';
  file: string;
  loaded: number;
  total: number;
}

export interface TokenMsg {
  type: 'token';
  token: string;
}

/**
 * Embeddings captured from the last LIV (double-gated convolution) layer —
 * layer 9 of 10 (0-indexed), immediately before the first GQA block.
 *
 * Shape: [batch=1, seq_len, hidden_dim]
 *
 * This data is the input to the quaternary quantization (Q²) WASM kernel
 * (src/q2.wat).  The kernel mean-pools over seq_len, L2-normalises the
 * resulting n-dimensional vector, and produces a packed Uint8Array of n/4
 * Gray-encoded bytes together with a 64-bit transition key.
 */
export interface EmbeddingMsg {
  type: 'embedding';
  /**
   * Transferable buffer containing activation values laid out as
   * [seq_len × hidden_dim] in row-major order.
   *
   * Element width depends on `dtype`:
   *   fp32 — 4 bytes, IEEE 754 single-precision (default)
   *   fp16 — 2 bytes, IEEE 754 half-precision
   *   q8   — 1 byte,  signed int8 ∈ [−128, 127]
   *   q4   — ½ byte,  two unsigned nibbles per byte ∈ [0, 15]
   *   q2   — ¼ byte,  four 2-bit Z₄ symbols per byte (prior Q² pass)
   *
   * The sender should transfer this buffer via postMessage(msg, [data]) to
   * avoid structured-clone copying; the receiver constructs the appropriate
   * typed-array view based on `dtype`.
   */
  data: ArrayBuffer;
  seqLen: number;
  hiddenDim: number;
  /**
   * Native element dtype of the activation values in `data`.
   * Matches the dtype string used to load the ONNX model.
   * Defaults to 'fp32' when the ONNX runtime returns full-precision tensors
   * regardless of model weight quantisation level.
   */
  dtype: 'fp32' | 'fp16' | 'q8' | 'q4' | 'q2';
}

export interface DoneMsg {
  type: 'done';
}

export interface ErrorMsg {
  type: 'error';
  message: string;
}

export type WorkerOutMsg =
  | StatusMsg
  | ProgressMsg
  | TokenMsg
  | EmbeddingMsg
  | DoneMsg
  | ErrorMsg;

// ─── Shared data types ───────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface GenerationConfig {
  max_new_tokens: number;
  temperature: number;
  repetition_penalty: number;
}
