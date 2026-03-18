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
  Dtype,
} from './types.js';

/** Subset of the ProgressInfo union we care about for download tracking. */
interface DownloadProgress {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
}

/**
 * Typed wrapper for calling the transformers.js pipeline factory.
 *
 * The public `pipeline` overloads don't expose `device`, `progress_callback`,
 * or other runtime options we rely on, so we define the concrete shape we need
 * and cast once at the call-site boundary.
 */
interface PipelineOptions {
  dtype: Dtype;
  device: string;
  progress_callback: (progress: DownloadProgress) => void;
}

/**
 * Call signature we use when invoking the loaded TextGenerationPipeline.
 *
 * The runtime supports additional undocumented options (hidden_states, streamer,
 * stopping_criteria) that aren't in the public TypeScript overloads.
 */
interface PipelineCallable {
  (messages: ChatMessage[], options: GenerationCallOptions): Promise<GenerationOutput>;
}

/** Options passed to the pipeline at inference time. */
interface GenerationCallOptions {
  max_new_tokens: number;
  temperature: number;
  do_sample: boolean;
  repetition_penalty: number;
  streamer: TextStreamer;
  stopping_criteria: InterruptableStoppingCriteria | null;
  output_hidden_states: boolean;
}

/** Concrete shape of a single layer's output tensor from transformers.js. */
interface TensorLike {
  data?: Float32Array;
  dims?: number[];
}

/** Subset of the generation output shape we inspect for embedding extraction. */
interface GenerationOutput {
  hidden_states?: TensorLike[][];
}

// ─── In a Web Worker, `self` is DedicatedWorkerGlobalScope. ─────────────────
// TypeScript's DOM lib types the global `self` as Window & typeof globalThis,
// so an explicit cast is required to access worker-specific methods.
const workerScope = self as unknown as DedicatedWorkerGlobalScope;

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
  workerScope.postMessage(msg, transfer);
}

// ─── Model loading ────────────────────────────────────────────────────────────

/**
 * Returns the EmbeddingMsg dtype used by the Q² kernel.
 *
 * Note: transformers.js typically returns hidden-state tensors as Float32Array
 * (fp32) even when model weights are quantised (q4/q8/fp16). Since the
 * extracted activations are currently handled as fp32 and forwarded as raw
 * bytes without conversion, we must advertise them as 'fp32' to avoid the
 * main thread / Q² kernel misinterpreting the buffer with the wrong element
 * width.
 *
 * If the runtime is ever extended to expose sub-fp32 activation tensors, this
 * function should be updated to derive the dtype from the actual tensor data
 * type rather than the model weight dtype.
 */
function toEmbeddingDtype(_dtype: Dtype): EmbeddingMsg['dtype'] {
  // The dtype argument is currently unused because transformers.js returns fp32
  // hidden states regardless of model weight quantisation.
  void _dtype;
  return 'fp32';
}

async function loadModel(modelId: string, dtype: Dtype, apiToken?: string): Promise<void> {
  // Apply the API token before any network requests (used by the Hub client).
  if (apiToken) {
    // env.accessToken is a runtime-mutable property used by the HF Hub client;
    // it is not reflected in the published TypeScript types.
    Object.assign(env, { accessToken: apiToken });
  }

  send({ type: 'status', status: 'loading', detail: 'Fetching model weights…' });

  // Cast the polymorphic `pipeline` factory to the concrete signature we need.
  // The public overloads omit runtime options (device, progress_callback) that
  // the underlying JS implementation supports.
  type PipelineFactory = (
    task: string,
    model: string,
    options: PipelineOptions,
  ) => Promise<TextGenerationPipeline>;
  const loadPipeline = pipeline as unknown as PipelineFactory;

  let lastErr: unknown;
  for (const device of DEVICE_PRIORITY) {
    try {
      send({ type: 'status', status: 'loading', detail: `Trying ${device.toUpperCase()} backend…` });

      pipe = await loadPipeline('text-generation', modelId, {
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
      });

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
    // Only request hidden states (expensive) when embeddings are explicitly
    // requested by the caller via config.return_embeddings.
    const extConfig = config as GenerationConfig & { return_embeddings?: boolean };
    const wantEmbeddings = extConfig.return_embeddings === true;

    // Cast the loaded pipeline to our typed callable interface.
    // The underlying JS pipeline supports these runtime options; they are not
    // reflected in the published TypeScript overloads.
    const callPipeline = pipe as unknown as PipelineCallable;
    const output = await callPipeline(messages, {
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
      output_hidden_states: wantEmbeddings,
    });

    if (wantEmbeddings) {
      // Extract the last-step hidden state from the completed generation.
      // hidden_states is TensorLike[][] — [generation_step][layer_index].
      const hiddenStates = output.hidden_states;
      if (Array.isArray(hiddenStates) && hiddenStates.length > 0) {
        // Take the last generation step; layer 9 = last LIV block output.
        const lastStep = hiddenStates[hiddenStates.length - 1];
        const lastConvOut: TensorLike | undefined = lastStep?.[9];
        if (lastConvOut?.data && lastConvOut.dims) {
          const [, seqLen, hiddenDim] = lastConvOut.dims;
          if (seqLen !== undefined && hiddenDim !== undefined) {
            // Transfer the buffer to avoid structured-clone copying.
            // The activation tensor buffer is always a regular ArrayBuffer
            // (transformers.js uses Float32Array, not SharedArrayBuffer).
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

workerScope.addEventListener(
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
