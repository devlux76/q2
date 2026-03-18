/**
 * worker.ts — Inference Web Worker
 *
 * Runs any Hugging Face ONNX text-generation model inside a dedicated Web
 * Worker so that heavy computation never blocks the UI thread.
 *
 * Device fallback order: WebNN → WebGPU → WebGL → WASM
 *
 * Architecture notes (LFM2.5-1.2B, when that model is selected):
 *   ┌─────────────────────────────────────────────────────────┐
 *   │  10 × LIV blocks  (double-gated 1-D causal convolution) │
 *   │   └─ Layer 9 output (0-based) → [last conv embeddings]  │
 *   │  6  × GQA blocks  (Grouped Query Attention)             │
 *   │  Output head                                            │
 *   └─────────────────────────────────────────────────────────┘
 *
 * The embeddings from layer 9 are the input to the quaternary quantisation
 * (Q²) WASM kernel that will be wired in when the .wat file is available.
 */

import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
  env,
  type TextGenerationPipeline,
} from '@huggingface/transformers';
import type {
  WorkerInMsg,
  WorkerOutMsg,
  ChatMessage,
  GenerationConfig,
  EmbeddingMsg,
} from './types.js';

/** Subset of the ProgressInfo union we care about for download tracking. */
interface DownloadProgress {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
}

// ─── Configuration ────────────────────────────────────────────────────────────

/**
 * dtype: 'q4' → loads model_q4.onnx
 * Most onnx-community models ship a q4-quantised file.
 * The dtype is now passed dynamically from the load message.
 */

/**
 * Preferred inference backends in priority order.
 * The worker tries each one in turn and settles on the first that succeeds.
 *   webnn  – Web Neural Network API (hardware-accelerated where available)
 *   webgpu – GPU-accelerated via the WebGPU API
 *   webgl  – Fallback GPU path via WebGL
 *   wasm   – Pure WebAssembly (always available)
 */
const DEVICE_PRIORITY = ['webnn', 'webgpu', 'webgl', 'wasm'] as const;
type ModelDevice = (typeof DEVICE_PRIORITY)[number];

// ─── Environment ──────────────────────────────────────────────────────────────

// Only fetch from HuggingFace Hub; disable Node-FS cache (we're in a browser).
env.allowLocalModels = false;
env.useBrowserCache = true;   // Cache model shards in the Cache API (IndexedDB)

// ─── State ────────────────────────────────────────────────────────────────────

let pipe: TextGenerationPipeline | null = null;
let stoppingCriteria: InterruptableStoppingCriteria | null = null;
/** Activation dtype of the currently-loaded model. Forwarded in EmbeddingMsg. */
let activeDtype: EmbeddingMsg['dtype'] = 'fp32';

// ─── Messaging helpers ────────────────────────────────────────────────────────

function send(msg: WorkerOutMsg, transfer: Transferable[] = []): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg, transfer);
}

// ─── Model loading ────────────────────────────────────────────────────────────

/**
 * Maps an ONNX dtype string to the EmbeddingMsg dtype used by the Q² kernel.
 * The ONNX runtime (transformers.js) typically returns hidden-state tensors as
 * fp32 even when model weights are quantised (q4/q8/fp16), so 'fp32' is the
 * safe default.  Override only if the runtime exposes sub-fp32 activation
 * tensors for the loaded dtype.
 */
function toEmbeddingDtype(dtype: string): EmbeddingMsg['dtype'] {
  if (dtype === 'fp16') return 'fp16';
  if (dtype === 'q8')   return 'q8';
  if (dtype === 'q4')   return 'fp32'; // q4 weights, but activations are fp32
  return 'fp32';
}

async function loadModel(modelId: string, dtype: string, apiToken?: string): Promise<void> {
  // Apply the API token before any network requests (used by the Hub client).
  if (apiToken) {
    (env as unknown as Record<string, unknown>).accessToken = apiToken;
  }

  send({ type: 'status', status: 'loading', detail: 'Fetching model weights…' });

  let lastErr: unknown;
  for (const device of DEVICE_PRIORITY) {
    try {
      send({ type: 'status', status: 'loading', detail: `Trying ${device.toUpperCase()} backend…` });
       
      pipe = (await (pipeline as any)('text-generation', modelId, {
        dtype,
        device,
        progress_callback: (p: DownloadProgress) => {
          send({
            type: 'progress',
            file: p.file ?? '',
            loaded: p.loaded ?? 0,
            total: p.total ?? 0,
          });
        },
      })) as TextGenerationPipeline;

      // Record the embedding dtype for the loaded model so EmbeddingMsg is typed correctly.
      activeDtype = toEmbeddingDtype(dtype);

      send({ type: 'status', status: 'ready' });
      return;
    } catch (err) {
      lastErr = err;
      // Fall through to the next device in the priority list.
    }
  }

  send({ type: 'error', message: String(lastErr) });
}

// ─── Inference ────────────────────────────────────────────────────────────────

async function generateResponse(
  messages: ChatMessage[],
  config: GenerationConfig,
): Promise<void> {
  if (!pipe) {
    send({ type: 'error', message: 'Model not loaded. Call load first.' });
    return;
  }

  send({ type: 'status', status: 'generating' });
  stoppingCriteria = new InterruptableStoppingCriteria();

  // TextStreamer pushes decoded token text fragments to the main thread.
  const streamer = new TextStreamer(pipe.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => {
      send({ type: 'token', token: text });
    },
  });

  try {
    // ── Text generation ──────────────────────────────────────────────────────
    const output = await (pipe as unknown as (
      msgs: ChatMessage[],
      opts: Record<string, unknown>,
    ) => Promise<Record<string, unknown>>)(messages, {
      max_new_tokens: config.max_new_tokens,
      temperature: config.temperature,
      do_sample: config.temperature > 0,
      repetition_penalty: config.repetition_penalty,
      streamer,
      stopping_criteria: stoppingCriteria,
      // ── Embedding hook ───────────────────────────────────────────────────
      // output_hidden_states enables per-step hidden-state capture.
      // The hidden state at index 9 (0-based) is the output of the last LIV
      // convolution block (layer 9 of 10), just before the first GQA block —
      // the correct extraction point for the Q² kernel (DESIGN.md §1.5).
      output_hidden_states: true,
    });

    // Extract the last-step hidden state from the completed generation.
    // hidden_states is Tensor[][] — [generation_step][layer_index].
    const hiddenStates = (output as { hidden_states?: unknown[] }).hidden_states;
    if (Array.isArray(hiddenStates) && hiddenStates.length > 0) {
      // Take the last generation step; layer 9 = last LIV block output.
      const lastStep = hiddenStates[hiddenStates.length - 1] as unknown[];
      const lastConvOut = lastStep?.[9] as { data?: Float32Array; dims?: number[] } | undefined;
      if (lastConvOut?.data && lastConvOut.dims) {
        const [, seqLen, hiddenDim] = lastConvOut.dims;
        if (seqLen !== undefined && hiddenDim !== undefined) {
          // Transfer the buffer to avoid structured-clone copying.
          const data = lastConvOut.data.buffer.slice(
            lastConvOut.data.byteOffset,
            lastConvOut.data.byteOffset + lastConvOut.data.byteLength,
          ) as ArrayBuffer;
          send(
            { type: 'embedding', data, seqLen, hiddenDim, dtype: activeDtype },
            [data],
          );
        }
      }
    }

    send({ type: 'done' });
  } catch (err) {
    const msg = String(err);
    // InterruptableStoppingCriteria sets interrupted = true; treat as user cancel.
    if (!stoppingCriteria?.interrupted) {
      send({ type: 'error', message: msg });
    } else {
      send({ type: 'done' });
    }
  } finally {
    stoppingCriteria = null;
    send({ type: 'status', status: 'idle' });
  }
}

// ─── Message router ───────────────────────────────────────────────────────────

(self as unknown as DedicatedWorkerGlobalScope).addEventListener(
  'message',
  (e: MessageEvent<WorkerInMsg>) => {
    const msg = e.data;
    switch (msg.type) {
      case 'load':
        void loadModel(msg.modelId, msg.dtype, msg.apiToken);
        break;
      case 'generate':
        void generateResponse(msg.messages, msg.config);
        break;
      case 'abort':
        stoppingCriteria?.interrupt();
        break;
    }
  },
);
