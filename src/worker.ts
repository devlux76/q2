/**
 * worker.ts — Inference Web Worker
 *
 * Runs any Hugging Face ONNX text-generation model inside a dedicated Web
 * Worker so that heavy computation never blocks the UI thread.
 *
 * Device fallback order (default): WebNN → WebGPU → WebGL → WASM
 * On iOS / iPadOS, where WebNN is unavailable and WebGPU currently hangs,
 * the worker skips those backends and falls back to: WebGL → WASM.
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
 * Factory type used when casting the polymorphic `pipeline` function.
 * The public overloads omit runtime options (device, progress_callback) that
 * the underlying JS implementation supports.
 */
type PipelineFactory = (
  task: string,
  model: string,
  options: PipelineOptions,
) => Promise<TextGenerationPipeline>;

/**
 * Call signature we use when invoking the loaded TextGenerationPipeline.
 *
 * The runtime supports additional options (streamer, stopping_criteria)
 * that are not reflected in the published TypeScript type signatures but are
 * supported at runtime by the underlying JS implementation.
 *
 * Return type: the text-generation pipeline always returns decoded text as
 * TextGenerationSingle[] = [{ generated_text: string | ChatMessage[] }].
 * It does NOT return a model-output object — hidden_states are NOT available
 * via this interface.  See NOTE below for the correct embedding approach.
 */
interface PipelineCallable {
  (messages: ChatMessage[], options: GenerationCallOptions): Promise<TextGenerationSingle[]>;
}

/**
 * Actual output shape of the transformers.js text-generation pipeline.
 *
 * When the input is a Chat (array of messages), `generated_text` is the full
 * conversation including the new assistant turn.  The pipeline decodes token
 * IDs to text before returning — raw model outputs (logits, hidden states) are
 * NOT exposed through this interface.
 *
 * Reference: TextGenerationSingle typedef in @huggingface/transformers/types/pipelines.d.ts
 */
interface TextGenerationSingle {
  generated_text: string | ChatMessage[];
}

/** Options passed to the pipeline at inference time. */
interface GenerationCallOptions {
  max_new_tokens: number;
  temperature: number;
  do_sample: boolean;
  repetition_penalty: number;
  streamer: TextStreamer;
  stopping_criteria: InterruptableStoppingCriteria | null;
}

// NOTE: Embedding extraction via the text-generation pipeline is NOT supported.
// The text-generation pipeline returns decoded text (TextGenerationSingle[]), not
// raw model outputs.  Hidden states require a separate feature-extraction pipeline
// or a direct call to pipe.model.forward() with a model that exports per-layer
// hidden states in its ONNX graph (non-standard; standard onnx-community models
// export {logits, past_key_values} only).

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
 * Returns true when running on iOS / iPadOS.
 *
 * Apple requires all browsers on iOS to use WebKit, so this covers Safari,
 * Chrome, Firefox, and every other iOS browser.  WebNN is not implemented on
 * iOS WebKit, and WebGPU (present in Safari 17+) hangs indefinitely when
 * loading ONNX models instead of throwing a catchable error.  Skipping both
 * lets the backend loop fall straight through to WebGL → WASM.
 */
function isIOS(): boolean {
  const ua = navigator.userAgent;
  // Standard iOS devices; also catches iPadOS in "Request Desktop Website" mode
  // where navigator.platform reports "MacIntel" but touch points reveal a touchscreen.
  return /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

// Verbose debug logging for model loading and inference steps.
function workerLog(level: 'debug' | 'info' | 'warn' | 'error', message: string, ...args: unknown[]): void {
  const prefix = `[q2 worker] ${new Date().toISOString()} [${level}]`;
  if (level === 'debug') {
    console.debug(prefix, message, ...args);
  } else if (level === 'info') {
    console.info(prefix, message, ...args);
  } else if (level === 'warn') {
    console.warn(prefix, message, ...args);
  } else {
    console.error(prefix, message, ...args);
  }
}

/**
 * Preferred inference backends in priority order.
 * The worker tries each one in turn and settles on the first that succeeds.
 *   webnn  – Web Neural Network API (hardware-accelerated where available)
 *   webgpu – GPU-accelerated via the WebGPU API
 *   webgl  – Fallback GPU path via WebGL
 *   wasm   – Pure WebAssembly (always available)
 *
 * On iOS, WebNN is absent and WebGPU hangs without throwing, so we skip
 * both and start directly from WebGL.
 */
type BackendName = 'webnn' | 'webgpu' | 'webgl' | 'wasm';

const DEVICE_PRIORITY: readonly BackendName[] = isIOS()
  ? ['webgl', 'wasm']
  : ['webnn', 'webgpu', 'webgl', 'wasm'];

/**
 * Milliseconds of silence (no progress callback) after which a backend is
 * considered to have hung during ONNX session initialisation.
 *
 * Some backends — including WebGPU on iOS WebKit and WebNN on Chrome for
 * Linux — stall indefinitely instead of throwing a catchable error.  The
 * hang guard below races the loader against this idle threshold so the
 * backend loop can fall through to the next candidate automatically.
 *
 * Download traffic continuously resets the clock via the progress_callback,
 * so only a true post-download hang (no activity at all) will trigger it.
 */
const BACKEND_HANG_TIMEOUT_MS = 30_000;

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
  workerLog('info', 'loadModel invoked', { modelId, dtype, hasApiToken: Boolean(apiToken) });

  // Apply the API token before any network requests (used by the Hub client).
  if (apiToken) {
    // env.accessToken is a runtime-mutable property used by the HF Hub client;
    // it is not reflected in the published TypeScript types.
    workerLog('debug', 'Setting env.accessToken for HuggingFace Hub request');
    Object.assign(env, { accessToken: apiToken });
  }

  workerLog('info', 'Sending initial loading status message');
  send({ type: 'status', status: 'loading', detail: 'Fetching model weights…' });

  // Cast the polymorphic `pipeline` factory to the concrete signature we need.
  // The public overloads omit runtime options (device, progress_callback) that
  // the underlying JS implementation supports.
  const loadPipeline = pipeline as unknown as PipelineFactory;

  let lastErr: unknown;
  for (const device of DEVICE_PRIORITY) {
    try {
      workerLog('info', 'Attempting backend', { device });
      send({ type: 'status', status: 'loading', detail: `Trying ${device.toUpperCase()} backend…` });

      // Track the last time a progress event was received so the hang guard
      // can distinguish live downloads from a stalled session init.
      let lastProgressAt = Date.now();

      const loader = loadPipeline('text-generation', modelId, {
        dtype,
        device,
        progress_callback: (p: DownloadProgress) => {
          lastProgressAt = Date.now();
          const detail = { file: p.file ?? '', loaded: p.loaded ?? 0, total: p.total ?? 0 };
          workerLog('debug', 'progress callback', detail);
          send({
            type: 'progress',
            file: p.file ?? '',
            loaded: p.loaded ?? 0,
            total: p.total ?? 0,
          });
        },
      });

      // Hang guard: some backends (e.g. WebGPU on iOS, WebNN on Chrome/Linux)
      // stall forever during ONNX session init without throwing.  Race the
      // loader against an idle-timeout so the loop can fall through.
      const hangGuard = new Promise<never>((_, reject) => {
        const id = setInterval(() => {
          const ageMs = Date.now() - lastProgressAt;
          workerLog('debug', 'hang guard tick', { device, ageMs, timeoutMs: BACKEND_HANG_TIMEOUT_MS });
          if (ageMs >= BACKEND_HANG_TIMEOUT_MS) {
            clearInterval(id);
            const message = `${device} backend timed out after ${BACKEND_HANG_TIMEOUT_MS / 1000}s of inactivity`;
            workerLog('warn', message);
            reject(new Error(message));
          }
        }, 1_000);
        // Cancel the watchdog as soon as the loader settles (success or error)
        // so the interval cannot fire after the race is already decided.
        void loader.then(() => {
          workerLog('debug', 'loader settled, clearing hang guard interval', { device });
          clearInterval(id);
        }, (e) => {
          workerLog('debug', 'loader rejected, clearing hang guard interval', { device, error: e });
          clearInterval(id);
        });
      });

      pipe = await Promise.race([loader, hangGuard]);
      workerLog('info', 'Model loaded successfully', { device, modelId, dtype });

      // Record the embedding dtype for the loaded model so EmbeddingMsg is typed correctly.
      activeDtype = toEmbeddingDtype(dtype);
      workerLog('info', 'active dtype set after load', { activeDtype });

      send({ type: 'status', status: 'ready' });
      return;
    } catch (err) {
      lastErr = err;
      workerLog('warn', 'Backend failed', { device, error: err });
      // Summarise the failure for the loading screen so the user can see
      // which backend was skipped and why before the next attempt starts.
      const maxLen = 80;
      const reason = err instanceof Error ? err.message : String(err);
      const shortReason = reason.length > maxLen ? reason.slice(0, maxLen - 3) + '…' : reason;
      send({
        type: 'status',
        status: 'loading',
        detail: `${device.toUpperCase()} unavailable — ${shortReason}`,
      });
      // Fall through to the next device in the priority list.
    }
  }
  workerLog('error', 'All backends failed, sending error message to main thread', { lastErr });
  send({ type: 'error', message: String(lastErr) });
}

// ─── Inference ────────────────────────────────────────────────────────────────

async function generateResponse(
  messages: ChatMessage[],
  config: GenerationConfig,
): Promise<void> {
  workerLog('info', 'generateResponse invoked', {
    messagesCount: messages.length,
    config,
    modelLoaded: Boolean(pipe),
    activeDtype,
  });

  if (!pipe) {
    workerLog('error', 'generateResponse called before model load');
    send({ type: 'error', message: 'Model not loaded. Call load first.' });
    return;
  }

  workerLog('info', 'Starting generation', {
    messagesCount: messages.length,
    hasConfig: Boolean(config),
  });
  send({ type: 'status', status: 'generating' });
  stoppingCriteria = new InterruptableStoppingCriteria();

  // TextStreamer pushes decoded token text fragments to the main thread.
  // Per-token logging is disabled by default: the callback runs on every
  // streamed chunk and console I/O in a Worker can measurably hurt throughput.
  // Flip to true locally when debugging token-level output.
  const DEBUG_TOKEN_LOGGING = false;
  let tokenCount = 0;
  const streamer = new TextStreamer(pipe.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => {
      tokenCount++;
      if (DEBUG_TOKEN_LOGGING) {
        workerLog('debug', 'token streamed', { tokenIndex: tokenCount, tokenLength: text.length });
      }
      send({ type: 'token', token: text });
    },
  });

  try {
    // ── Text generation ──────────────────────────────────────────────────────
    // Cast the loaded pipeline to our typed callable interface.
    // The underlying JS pipeline supports these runtime options (streamer,
    // stopping_criteria) even though they are not in the published TypeScript
    // overloads.
    const callPipeline = pipe as unknown as PipelineCallable;
    const output = await callPipeline(messages, {
      max_new_tokens: config.max_new_tokens,
      temperature: config.temperature,
      do_sample: config.temperature > 0,
      repetition_penalty: config.repetition_penalty,
      streamer,
      stopping_criteria: stoppingCriteria,
    });

    // output is TextGenerationSingle[] = [{ generated_text: string | ChatMessage[] }]
    // The pipeline decodes token IDs to text and returns the full conversation.
    // Raw model outputs (logits, hidden states) are NOT available here.
    workerLog('info', 'Pipeline generation finished', {
      tokenCount,
      outputLength: output.length,
    });

    // ── Embedding extraction ─────────────────────────────────────────────────
    // NOTE: Accessing per-layer hidden states during generation is NOT
    // supported by the transformers.js text-generation pipeline.  The
    // model.generate() loop in transformers.js v3 does not collect hidden
    // states — output_hidden_states in GenerationConfig has no effect.
    //
    // The correct approach to obtain hidden states is:
    //   1. Use a feature-extraction pipeline with a dedicated embedding model.
    //   2. Or call pipe.model.forward() on the generated token sequence with
    //      a model that exports intermediate hidden states in its ONNX graph.
    //
    // Standard onnx-community text-generation models export only
    // {logits, past_key_values}.  To use Q² fingerprinting, configure a
    // dedicated embedding model via the benchModelT3 setting.
    const extConfig = config as GenerationConfig & { return_embeddings?: boolean };
    const wantEmbeddings = extConfig.return_embeddings === true;
    if (wantEmbeddings) {
      workerLog('warn',
        'Embedding extraction via text-generation pipeline is not supported. ' +
        'Use a feature-extraction pipeline with a dedicated embedding model instead.');
    }

    send({ type: 'done' });
  } catch (err) {
    const msg = String(err);
    workerLog('error', 'Generation failed', { error: err, message: msg });
    // InterruptableStoppingCriteria sets interrupted = true; treat as user cancel.
    if (!stoppingCriteria?.interrupted) {
      send({ type: 'error', message: msg });
    } else {
      workerLog('info', 'Generation aborted by user', { message: msg });
      send({ type: 'done' });
    }
  } finally {
    stoppingCriteria = null;
    workerLog('info', 'Generation completed/finalized, setting status idle');
    send({ type: 'status', status: 'idle' });
  }
}

// ─── Message router ───────────────────────────────────────────────────────────

workerScope.addEventListener(
  'message',
  (e: MessageEvent<WorkerInMsg>) => {
    const msg = e.data;
    workerLog('debug', 'Worker received message', { type: msg.type });
    switch (msg.type) {
      case 'load':
        workerLog('info', 'Worker message router: dispatching load', { modelId: msg.modelId, dtype: msg.dtype });
        void loadModel(msg.modelId, msg.dtype, msg.apiToken);
        break;
      case 'generate':
        workerLog('info', 'Worker message router: dispatching generate', { messagesCount: msg.messages.length });
        void generateResponse(msg.messages, msg.config);
        break;
      case 'abort':
        workerLog('info', 'Worker message router: dispatching abort');
        stoppingCriteria?.interrupt();
        break;
    }
  },
);
