// Shared message types between the main thread and the inference Web Worker.
// All communication uses structured-clone-compatible payloads.

// ─── Main thread → Worker ────────────────────────────────────────────────────

export interface LoadModelMsg {
  type: 'load';
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
 * This data is the input to the quaternary quantization (Q²) WASM kernel.
 * The kernel receives this data as a Float32Array view over the provided
 * ArrayBuffer and writes back a compact Uint8Array where each byte encodes
 * four 2-bit quaternary symbols.
 */
export interface EmbeddingMsg {
  type: 'embedding';
  /**
   * Transferable buffer containing float32 values laid out as
   * [seq_len × hidden_dim]. The sender should transfer this buffer via
   * postMessage(msg, [data]) to avoid structured-clone copying, and the
   * receiver should construct a Float32Array view as needed.
   */
  data: ArrayBuffer;
  seqLen: number;
  hiddenDim: number;
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
