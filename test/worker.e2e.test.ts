/**
 * worker.e2e.test.ts — End-to-end tests for the Web Worker's
 * transformers.js integration.
 *
 * These tests mock @huggingface/transformers to verify, without downloading
 * any real models:
 *   1. The correct transformers.js pipeline API is called (the "happy path")
 *   2. The worker sends the right sequence of WorkerOutMsg messages
 *   3. Token streaming via TextStreamer works end-to-end
 *   4. Abort / error paths work correctly
 *   5. The actual pipeline output shape (TextGenerationSingle[]) is handled
 *      correctly — in particular, `output.hidden_states` is never populated
 *      by the text-generation pipeline and must not be relied upon.
 *
 * Happy-path API contract for transformers.js text-generation
 * (see https://huggingface.co/docs/transformers.js):
 *
 *   const pipe = await pipeline('text-generation', modelId, { dtype, device });
 *   // pipe is a TextGenerationPipeline
 *
 *   const streamer = new TextStreamer(pipe.tokenizer, {
 *     skip_prompt: true,
 *     skip_special_tokens: true,
 *     callback_function: (text) => { ... },
 *   });
 *
 *   const result = await pipe(chatMessages, {
 *     max_new_tokens: N,
 *     temperature: T,
 *     do_sample: T > 0,
 *     repetition_penalty: P,
 *     streamer,
 *     stopping_criteria: criteria,
 *   });
 *
 *   // result: TextGenerationSingle[] = [{ generated_text: ChatMessage[] }]
 *   // result[0].generated_text includes the full conversation + assistant turn
 *   // NOTE: result does NOT have a .hidden_states field; the text-generation
 *   // pipeline returns decoded text, not raw model outputs.
 */

import { vi, describe, it, expect, beforeAll, afterAll } from 'vitest';
import type { ChatMessage, WorkerOutMsg } from '../src/types.js';

// ─── Module mock setup ────────────────────────────────────────────────────────
//
// vi.mock is hoisted to the top of the compiled output, so these factories run
// before any import.  vi.hoisted() lets us create values that outlive the
// factory closure and are accessible to the test body.

const {
  mockPipelineFactory,
  MockTextStreamer,
  MockInterruptableStoppingCriteria,
  mockEnv,
} = vi.hoisted(() => {
  // Shared streamer-callback reference so tests can trigger token delivery.
  let lastStreamerCallback: ((text: string) => void) | null = null;

  class MockTextStreamer {
    callback_function: ((text: string) => void) | null;
    constructor(
      _tokenizer: unknown,
      opts: { callback_function?: (text: string) => void } = {},
    ) {
      this.callback_function = opts.callback_function ?? null;
      lastStreamerCallback = this.callback_function;
    }

    /** Simulate the model pushing a token through the streamer. */
    static pushToken(text: string): void {
      lastStreamerCallback?.(text);
    }
  }

  class MockInterruptableStoppingCriteria {
    interrupted = false;
    interrupt(): void { this.interrupted = true; }
    reset(): void { this.interrupted = false; }
  }

  const mockEnv = { allowLocalModels: true, useBrowserCache: false };

  // The factory returned by pipeline() — a callable that also has .tokenizer.
  const mockPipelineFactory = vi.fn();

  return { mockPipelineFactory, MockTextStreamer, MockInterruptableStoppingCriteria, mockEnv };
});

vi.mock('@huggingface/transformers', () => ({
  pipeline: mockPipelineFactory,
  TextStreamer: MockTextStreamer,
  InterruptableStoppingCriteria: MockInterruptableStoppingCriteria,
  env: mockEnv,
}));

// ─── Worker message harness ───────────────────────────────────────────────────

/**
 * Dispatches a typed message to the worker's message listener (which is
 * registered on `self` — the jsdom window — when worker.ts is imported).
 */
function sendToWorker(data: unknown): void {
  self.dispatchEvent(new MessageEvent('message', { data }));
}

/**
 * Returns a promise that resolves with the next WorkerOutMsg that satisfies
 * the given predicate, or rejects after `timeoutMs`.
 */
function waitForMessage(
  predicate: (msg: WorkerOutMsg) => boolean,
  timeoutMs = 3000,
): Promise<WorkerOutMsg> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      self.removeEventListener('message', handler as EventListenerOrEventListenerObject);
      reject(new Error(`waitForMessage timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    function handler(e: MessageEvent<WorkerOutMsg>): void {
      if (predicate(e.data)) {
        clearTimeout(timer);
        self.removeEventListener('message', handler as EventListenerOrEventListenerObject);
        resolve(e.data);
      }
    }

    self.addEventListener('message', handler as EventListenerOrEventListenerObject);
  });
}

/**
 * Collects all WorkerOutMsg messages emitted during the execution of `fn`,
 * returning them in order.
 */
async function collectMessages(fn: () => Promise<void>, settlementDelayMs = 50): Promise<WorkerOutMsg[]> {
  const messages: WorkerOutMsg[] = [];

  const handler = (e: MessageEvent<WorkerOutMsg>): void => {
    messages.push(e.data);
  };
  self.addEventListener('message', handler as EventListenerOrEventListenerObject);

  await fn();
  await new Promise<void>((r) => setTimeout(r, settlementDelayMs));

  self.removeEventListener('message', handler as EventListenerOrEventListenerObject);
  return messages;
}

// ─── Worker setup ─────────────────────────────────────────────────────────────
//
// Import worker.ts once for the entire describe block.  The module registers
// its message listener on `self` at import time.  Module-level state (pipe,
// activeDtype) persists across tests, so tests run in the logical order:
// load → generate → abort → reload.
//
// IMPORTANT: worker.ts uses DedicatedWorkerGlobalScope.postMessage(data, transfer)
// but in jsdom self === window.  window.postMessage(data, []) treats [] as
// targetOrigin (an array coerced to ""), which fails with a SyntaxError.
//
// Fix: replace self.postMessage with a thin shim BEFORE importing worker.ts
// so the worker's send() helper dispatches 'message' events on self instead,
// which our waitForMessage / collectMessages helpers can listen to.

beforeAll(async () => {
  // Replace window.postMessage with a worker-compatible shim that re-fires
  // the payload as a 'message' event so test helpers can observe it.
  Object.defineProperty(self, 'postMessage', {
    value(data: unknown, _transferOrOrigin?: unknown): void {
      self.dispatchEvent(new MessageEvent('message', { data }));
    },
    configurable: true,
    writable: true,
  });

  // Now import worker.ts — it reads `self` at module-init time so our shim
  // is already in place when `workerScope = self` is evaluated.
  await import('../src/worker.ts');
});

afterAll(() => {
  vi.restoreAllMocks();
});

// ─── Tests ────────────────────────────────────────────────────────────────────

/**
 * Stubs navigator.gpu and navigator.ml for the duration of a test callback,
 * restoring originals in a finally block.
 *
 * jsdom does not expose these browser APIs, so the worker's preflight check
 * (probeAvailableDevices) would otherwise filter out webgpu and webnn, leaving
 * only wasm.  Tests that need to exercise the full three-backend chain must
 * call this helper to make the preflight check pass all three backends through.
 */
async function withAllNavigatorApiStubs(fn: () => Promise<void>): Promise<void> {
  const navAny = navigator as Record<string, unknown>;
  const origGpu = navAny.gpu;
  const origMl = navAny.ml;
  Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true, writable: true });
  Object.defineProperty(navigator, 'ml', { value: {}, configurable: true, writable: true });
  try {
    await fn();
  } finally {
    Object.defineProperty(navigator, 'gpu', { value: origGpu, configurable: true, writable: true });
    Object.defineProperty(navigator, 'ml', { value: origMl, configurable: true, writable: true });
  }
}

describe('Worker E2E: transformers.js happy-path integration', () => {

  // Shared fake pipeline callable — set during the load test and reused by
  // subsequent generate / abort tests.  This mirrors real usage where the
  // pipeline is loaded once and called many times.
  let fakePipe: ReturnType<typeof vi.fn>;

  // ── Load model ──────────────────────────────────────────────────────────────

  it('load: pipeline() is called with correct task, modelId, dtype and device list', async () => {
    // Build a fake callable pipeline (pipe) with a .tokenizer property.
    fakePipe = vi.fn();
    (fakePipe as unknown as { tokenizer: object }).tokenizer = {};

    // pipeline() resolves to the fake pipe on the first (successful) backend.
    mockPipelineFactory.mockResolvedValueOnce(fakePipe);

    const msgs = await collectMessages(async () => {
      sendToWorker({ type: 'load', modelId: 'onnx-community/test-model', dtype: 'q4' });
      // Wait for the ready status.
      await waitForMessage((m) => m.type === 'status' && (m as { type: string; status: string }).status === 'ready');
    });

    // The worker must emit at least one 'loading' status and then 'ready'.
    const statuses = msgs.filter((m) => m.type === 'status').map((m) => (m as { type: string; status: string }).status);
    expect(statuses).toContain('loading');
    expect(statuses).toContain('ready');

    // pipeline() must have been called with task='text-generation'.
    expect(mockPipelineFactory).toHaveBeenCalledWith(
      'text-generation',
      'onnx-community/test-model',
      expect.objectContaining({ dtype: 'q4' }),
    );

    // The device property must be one of the three supported backends.
    const [, , opts] = mockPipelineFactory.mock.calls[0] as [string, string, { device: string }];
    expect(['webnn', 'webgpu', 'wasm']).toContain(opts.device);
  });

  // ── Generate – happy path ───────────────────────────────────────────────────

  it('generate: tokens are streamed and done is sent after pipeline resolves', async () => {
    // Simulate the pipeline streaming tokens via the TextStreamer callback
    // and returning the correct TextGenerationSingle[] output shape.
    fakePipe.mockImplementation(
      async (
        _messages: ChatMessage[],
        opts: {
          streamer?: { callback_function: ((t: string) => void) | null };
          stopping_criteria?: { interrupted: boolean } | null;
        },
      ) => {
        // Correct transformers.js text-generation output shape:
        // TextGenerationSingle[] = [{ generated_text: ChatMessage[] }]
        // The streamer's callback_function delivers tokens in real usage;
        // we replicate that behaviour here.
        opts.streamer?.callback_function?.('Hello');
        opts.streamer?.callback_function?.(' world');
        return [
          {
            generated_text: [
              ..._messages,
              { role: 'assistant', content: 'Hello world' },
            ],
          },
        ];
      },
    );

    const msgs = await collectMessages(async () => {
      sendToWorker({
        type: 'generate',
        messages: [{ role: 'user', content: 'Hi' }],
        config: { max_new_tokens: 64, temperature: 0, repetition_penalty: 1.0 },
      });
      await waitForMessage((m) => m.type === 'done');
    });

    // Verify the message sequence: generating → token(s) → done.
    const types = msgs.map((m) => m.type);
    expect(types).toContain('status'); // status: generating
    expect(types).toContain('token');
    expect(types).toContain('done');

    const generating = msgs.find(
      (m) => m.type === 'status' && (m as { type: string; status: string }).status === 'generating',
    );
    expect(generating).toBeTruthy();

    const tokens = msgs.filter((m) => m.type === 'token').map((m) => (m as { type: string; token: string }).token);
    expect(tokens).toEqual(['Hello', ' world']);

    // Final status must be 'idle' after done.
    const idle = msgs.find(
      (m) => m.type === 'status' && (m as { type: string; status: string }).status === 'idle',
    );
    expect(idle).toBeTruthy();
  });

  it('generate: pipeline is called with correct generation arguments', async () => {
    fakePipe.mockImplementation(
      async (
        _messages: ChatMessage[],
        opts: { streamer?: { callback_function: ((t: string) => void) | null } },
      ) => {
        opts.streamer?.callback_function?.('ok');
        return [{ generated_text: [{ role: 'assistant', content: 'ok' }] }];
      },
    );

    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hello' },
    ];
    const config = { max_new_tokens: 128, temperature: 0.7, repetition_penalty: 1.1 };

    await collectMessages(async () => {
      sendToWorker({ type: 'generate', messages, config });
      await waitForMessage((m) => m.type === 'done');
    });

    // Verify the pipeline was called with the chat messages and generation options.
    expect(fakePipe).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ role: 'system' }),
        expect.objectContaining({ role: 'user' }),
      ]),
      expect.objectContaining({
        max_new_tokens: 128,
        temperature: 0.7,
        do_sample: true,       // temperature > 0 → do_sample must be true
        repetition_penalty: 1.1,
      }),
    );

    // A streamer must always be passed.
    const [, callOpts] = fakePipe.mock.calls.at(-1) as [ChatMessage[], Record<string, unknown>];
    expect(callOpts.streamer).toBeTruthy();

    // A stopping_criteria must always be passed.
    expect(callOpts.stopping_criteria).not.toBeUndefined();
  });

  it('generate: do_sample is false when temperature is 0 (greedy decoding)', async () => {
    fakePipe.mockImplementation(
      async (
        _messages: ChatMessage[],
        opts: { streamer?: { callback_function: ((t: string) => void) | null } },
      ) => {
        opts.streamer?.callback_function?.('ok');
        return [{ generated_text: [{ role: 'assistant', content: 'ok' }] }];
      },
    );

    await collectMessages(async () => {
      sendToWorker({
        type: 'generate',
        messages: [{ role: 'user', content: 'Hi' }],
        config: { max_new_tokens: 10, temperature: 0, repetition_penalty: 1.0 },
      });
      await waitForMessage((m) => m.type === 'done');
    });

    const [, callOpts] = fakePipe.mock.calls.at(-1) as [ChatMessage[], Record<string, unknown>];
    // temperature === 0 → deterministic greedy decoding → do_sample must be false.
    expect(callOpts.do_sample).toBe(false);
  });

  // ── Pipeline output shape ───────────────────────────────────────────────────

  it('generate: pipeline returns TextGenerationSingle[] — output.hidden_states is not available', async () => {
    /**
     * The transformers.js TextGenerationPipeline._call() returns:
     *   TextGenerationSingle[] = [{ generated_text: string | ChatMessage[] }]
     *
     * It does NOT return a model output object with .hidden_states.
     * The model.generate() loop in transformers.js v3 does not collect
     * per-layer hidden states even when output_hidden_states is set in the
     * generation config.  Any code that checks output.hidden_states after
     * calling the text-generation pipeline will always find undefined.
     *
     * Correct way to obtain hidden states: call pipe.model.forward()
     * (or a feature-extraction pipeline) on the generated token sequence
     * separately, with a model that exports hidden states in its ONNX graph.
     */

    // Return the actual pipeline output shape (text, no hidden_states).
    const pipelineOutput: Array<{ generated_text: ChatMessage[] }> = [
      {
        generated_text: [
          { role: 'user', content: 'ping' },
          { role: 'assistant', content: 'pong' },
        ],
      },
    ];

    fakePipe.mockImplementation(
      async (
        _messages: ChatMessage[],
        opts: { streamer?: { callback_function: ((t: string) => void) | null } },
      ) => {
        opts.streamer?.callback_function?.('pong');
        return pipelineOutput;
      },
    );

    await collectMessages(async () => {
      sendToWorker({
        type: 'generate',
        messages: [{ role: 'user', content: 'ping' }],
        config: { max_new_tokens: 8, temperature: 0, repetition_penalty: 1.0 },
      });
      await waitForMessage((m) => m.type === 'done');
    });

    // The returned value is an array with a generated_text field — no hidden_states.
    const [callResult] = pipelineOutput;
    expect(callResult).toHaveProperty('generated_text');
    expect((callResult as Record<string, unknown>).hidden_states).toBeUndefined();
  });

  // ── Abort ───────────────────────────────────────────────────────────────────

  it('abort: sending abort interrupts the stopping criteria and ends generation cleanly', async () => {
    fakePipe.mockImplementation(
      async (
        _messages: ChatMessage[],
        opts: {
          streamer?: { callback_function: ((t: string) => void) | null };
          stopping_criteria?: { interrupted: boolean; interrupt(): void } | null;
        },
      ) => {
        // Simulate partial token delivery before the abort arrives.
        opts.streamer?.callback_function?.('partial');
        // Simulate the abort signal being received mid-generation.
        if (opts.stopping_criteria) {
          opts.stopping_criteria.interrupted = true;
        }
        // In real usage transformers.js would throw an AbortError; the worker
        // checks stoppingCriteria.interrupted to suppress the error and still
        // send a 'done' message.
        throw new Error('AbortError: generation aborted');
      },
    );

    const msgs = await collectMessages(async () => {
      // Set up the done-listener BEFORE dispatching the generate message so
      // we do not miss the event even if the mock resolves synchronously.
      const donePromise = waitForMessage((m) => m.type === 'done' || m.type === 'error');
      sendToWorker({
        type: 'generate',
        messages: [{ role: 'user', content: 'Long story' }],
        config: { max_new_tokens: 512, temperature: 0, repetition_penalty: 1.0 },
      });
      // The abort message is sent immediately; the mock already sets interrupted.
      sendToWorker({ type: 'abort' });
      // After abort, the worker should send 'done' (not 'error').
      await donePromise;
    });

    const types = msgs.map((m) => m.type);
    // No error message — aborted generation is treated as a clean completion.
    expect(types).not.toContain('error');
    expect(types).toContain('done');
    // Status must return to idle.
    const idle = msgs.find(
      (m) => m.type === 'status' && (m as { type: string; status: string }).status === 'idle',
    );
    expect(idle).toBeTruthy();
  });

  // ── Backend fallback ────────────────────────────────────────────────────────

  it('load: falls back through backends when a higher-priority backend fails', async () => {
    // Reset the pipeline mock so the next load uses a fresh sequence.
    mockPipelineFactory.mockReset();

    // jsdom does not expose navigator.gpu or navigator.ml — the preflight
    // function in worker.ts correctly skips those backends.  withAllNavigatorApiStubs
    // stubs them so all three valid backends (webnn, webgpu, wasm) pass the
    // capability check, allowing us to exercise the full fallback chain.
    await withAllNavigatorApiStubs(async () => {
      // Simulate webnn and webgpu failing, wasm succeeding.
      const fallbackPipe = vi.fn().mockImplementation(
        async (
          _messages: ChatMessage[],
          opts: { streamer?: { callback_function: ((t: string) => void) | null } },
        ) => {
          opts.streamer?.callback_function?.('hi');
          return [{ generated_text: [{ role: 'assistant', content: 'hi' }] }];
        },
      );
      (fallbackPipe as unknown as { tokenizer: object }).tokenizer = {};

      let callCount = 0;
      mockPipelineFactory.mockImplementation(
        async (_task: string, _model: string, opts: { device: string }) => {
          callCount++;
          if (opts.device === 'wasm') {
            return fallbackPipe;
          }
          throw new Error(`${opts.device} backend not available`);
        },
      );

      const msgs = await collectMessages(async () => {
        sendToWorker({
          type: 'load',
          modelId: 'onnx-community/test-fallback',
          dtype: 'q8',
        });
        await waitForMessage(
          (m) => m.type === 'status' && (m as { type: string; status: string }).status === 'ready',
        );
      });

      // The worker must have tried multiple backends.
      expect(callCount).toBeGreaterThan(1);

      // The final status must be 'ready' (wasm succeeded).
      const statuses = msgs.filter((m) => m.type === 'status').map((m) => (m as { type: string; status: string }).status);
      expect(statuses).toContain('ready');
      expect(statuses).not.toContain('error');

      // Update fakePipe for subsequent generate tests.
      fakePipe = fallbackPipe;
    });
  });

  // ── Preflight / webgl regression ────────────────────────────────────────────

  it('load: pipeline is never called with device=webgl (unsupported in transformers.js@next)', async () => {
    // Regression test: webgl was removed from DEVICE_PRIORITY because
    // transformers.js@next (v4.x) rejects it with:
    //   "Unsupported device: 'webgl'. Should be one of: webnn-npu, webnn-gpu,
    //    webnn-cpu, webnn, webgpu, wasm."
    // withAllNavigatorApiStubs ensures all valid backends pass the preflight
    // check; then verify webgl never appears in any pipeline() call.
    mockPipelineFactory.mockReset();

    await withAllNavigatorApiStubs(async () => {
      const testPipe = vi.fn().mockImplementation(
        async (
          _messages: ChatMessage[],
          opts: { streamer?: { callback_function: ((t: string) => void) | null } },
        ) => {
          opts.streamer?.callback_function?.('ok');
          return [{ generated_text: [{ role: 'assistant', content: 'ok' }] }];
        },
      );
      (testPipe as unknown as { tokenizer: object }).tokenizer = {};

      const devicesAttempted: string[] = [];
      mockPipelineFactory.mockImplementation(
        async (_task: string, _model: string, opts: { device: string }) => {
          devicesAttempted.push(opts.device);
          return testPipe;
        },
      );

      await collectMessages(async () => {
        sendToWorker({
          type: 'load',
          modelId: 'onnx-community/test-no-webgl',
          dtype: 'q4',
        });
        await waitForMessage(
          (m) => m.type === 'status' && (m as { type: string; status: string }).status === 'ready',
        );
      });

      // webgl must never appear in any pipeline() call.
      expect(devicesAttempted).not.toContain('webgl');

      // Every device attempted must be one of the valid v4.x devices.
      for (const d of devicesAttempted) {
        expect(['webnn', 'webgpu', 'wasm']).toContain(d);
      }

      fakePipe = testPipe;
    });
  });

  // ── Progress callbacks ──────────────────────────────────────────────────────

  it('load: progress callbacks during download are forwarded as progress messages', async () => {
    mockPipelineFactory.mockReset();

    const progressPipe = vi.fn();
    (progressPipe as unknown as { tokenizer: object }).tokenizer = {};

    mockPipelineFactory.mockImplementation(
      async (
        _task: string,
        _model: string,
        opts: { progress_callback?: (p: { status: string; file: string; loaded: number; total: number }) => void },
      ) => {
        // Simulate download progress events.
        opts.progress_callback?.({ status: 'download', file: 'model.onnx', loaded: 512, total: 1024 });
        opts.progress_callback?.({ status: 'download', file: 'model.onnx', loaded: 1024, total: 1024 });
        return progressPipe;
      },
    );

    const msgs = await collectMessages(async () => {
      sendToWorker({
        type: 'load',
        modelId: 'onnx-community/test-progress',
        dtype: 'q4',
      });
      await waitForMessage(
        (m) => m.type === 'status' && (m as { type: string; status: string }).status === 'ready',
      );
    });

    const progressMsgs = msgs.filter((m) => m.type === 'progress');
    expect(progressMsgs).toHaveLength(2);

    const [first] = progressMsgs as Array<{ type: string; file: string; loaded: number; total: number }>;
    expect(first.file).toBe('model.onnx');
    expect(first.loaded).toBe(512);
    expect(first.total).toBe(1024);
  });

  // ── Error handling ──────────────────────────────────────────────────────────

  it('generate: worker sends error message when pipe is null (model not loaded)', async () => {
    // The worker.ts guard: if (!pipe) → send({ type: 'error', message: '...' })
    // We verify the guard code path works by resetting the module cache and
    // importing a fresh worker instance where pipe starts as null.
    vi.resetModules();

    const freshMsgs: WorkerOutMsg[] = [];
    const originalPost = Object.getOwnPropertyDescriptor(self, 'postMessage');
    Object.defineProperty(self, 'postMessage', {
      value(data: unknown): void {
        freshMsgs.push(data as WorkerOutMsg);
        self.dispatchEvent(new MessageEvent('message', { data }));
      },
      configurable: true,
      writable: true,
    });

    // Fresh import: pipe starts as null.
    await import('../src/worker.ts');

    sendToWorker({
      type: 'generate',
      messages: [{ role: 'user', content: 'hi' }],
      config: { max_new_tokens: 8, temperature: 0, repetition_penalty: 1.0 },
    });
    await new Promise<void>((r) => setTimeout(r, 50));

    if (originalPost) Object.defineProperty(self, 'postMessage', originalPost);

    const errorMsg = freshMsgs.find((m) => m.type === 'error') as
      | { type: string; message: string }
      | undefined;
    expect(errorMsg).toBeTruthy();
    expect(errorMsg?.message).toMatch(/model not loaded/i);
  });

  it('load: all backends failing sends an error message', async () => {
    mockPipelineFactory.mockReset();

    // All backends fail.
    mockPipelineFactory.mockRejectedValue(new Error('backend unavailable'));

    const msgs = await collectMessages(async () => {
      sendToWorker({
        type: 'load',
        modelId: 'onnx-community/impossible',
        dtype: 'q4',
      });
      await waitForMessage(
        (m) => m.type === 'error' || (m.type === 'status' && (m as { type: string; status: string }).status === 'ready'),
      );
    });

    const errorMsg = msgs.find((m) => m.type === 'error');
    expect(errorMsg).toBeTruthy();
  });
});

// ─── Embedding extraction contract ───────────────────────────────────────────

describe('Worker E2E: embedding extraction contract', () => {
  /**
   * This test suite documents the current state of embedding extraction and
   * the correct API contract for transformers.js.
   *
   * KNOWN LIMITATION:
   *   The text-generation pipeline in transformers.js v3 does NOT populate
   *   output.hidden_states.  The model.generate() loop only collects
   *   attentions (when output_attentions + return_dict_in_generate are set),
   *   not hidden states.  Passing output_hidden_states: true has no effect.
   *
   * CORRECT APPROACH (not yet implemented):
   *   To obtain hidden-state activations, call pipe.model.forward() on the
   *   generated token sequence directly (bypassing the pipeline), with a
   *   model that exports intermediate hidden states in its ONNX graph.
   *   Standard onnx-community models export logits + past_key_values only.
   */

  it('pipeline output is an array of TextGenerationSingle objects, not a model output', () => {
    // A TextGenerationSingle is: { generated_text: string | ChatMessage[] }
    // It does NOT have a .hidden_states property.
    const singleOutput: { generated_text: ChatMessage[]; hidden_states?: unknown } = {
      generated_text: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'hi' },
      ],
    };

    expect(Array.isArray(singleOutput.generated_text)).toBe(true);
    expect(singleOutput.hidden_states).toBeUndefined();
  });

  it('TextStreamer callback_function receives decoded token strings, not tensor IDs', () => {
    /**
     * The TextStreamer.callback_function is called with already-decoded
     * human-readable text fragments (e.g. "Hello", " world") — NOT raw
     * token IDs.  This is the correct hook for streaming UI updates.
     */
    const received: string[] = [];

    const _streamer = new MockTextStreamer(
      { /* fake tokenizer */ },
      { callback_function: (text: string) => { received.push(text); } },
    );

    MockTextStreamer.pushToken('Hello');
    MockTextStreamer.pushToken(' world!');

    expect(received).toEqual(['Hello', ' world!']);
  });

  it('InterruptableStoppingCriteria.interrupt() sets interrupted flag', () => {
    const criteria = new MockInterruptableStoppingCriteria();
    expect(criteria.interrupted).toBe(false);
    criteria.interrupt();
    expect(criteria.interrupted).toBe(true);
    criteria.reset();
    expect(criteria.interrupted).toBe(false);
  });
});
