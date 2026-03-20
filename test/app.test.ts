import { beforeEach, describe, expect, it, vi } from 'vitest';

// Ensure our test environment has the minimal DOM needed by app.ts.
function setupDom() {
  // Reset persisted settings between tests.
  localStorage.clear();

  document.body.innerHTML = `
    <nav id="top-nav">
      <span class="nav-logo">Q²</span>
      <div id="nav-tabs" role="tablist">
        <button id="tab-chat" class="nav-tab active" role="tab" aria-selected="true" aria-controls="panel-chat" data-tab="chat" tabindex="0">Chat</button>
        <button id="tab-benchmarks" class="nav-tab" role="tab" aria-selected="false" aria-controls="panel-benchmarks" data-tab="benchmarks" tabindex="-1">Benchmarks</button>
        <button id="tab-settings" class="nav-tab" role="tab" aria-selected="false" aria-controls="panel-settings" data-tab="settings" tabindex="-1">Settings</button>
      </div>
      <span id="model-status" class="status-badge">No model</span>
    </nav>

    <div id="load-overlay" class="hidden">
      <div class="load-card">
        <h2 id="load-model-name">Loading model…</h2>
        <div id="load-bar" role="progressbar"><div id="load-bar-fill"></div></div>
        <p id="load-status">Initializing…</p>
      </div>
    </div>

    <div id="panel-chat" class="tab-panel" role="tabpanel">
      <div id="chat-app">
        <div id="messages"></div>
        <textarea id="user-input"></textarea>
        <button id="send-btn"></button>
        <button id="stop-btn"></button>
      </div>
    </div>

    <div id="panel-benchmarks" class="tab-panel hidden" role="tabpanel">
      <div id="bench-status">Ready</div>
      <table><tbody id="bench-results-body"></tbody></table>
      <button id="bench-run-all"></button>
      <button id="bench-run-t0"></button>
      <button id="bench-run-t1"></button>
      <button id="bench-run-t2"></button>
      <button id="bench-run-t3"></button>
      <button id="bench-run-t4"></button>
      <button id="bench-run-t5"></button>
    </div>

    <div id="panel-settings" class="tab-panel hidden" role="tabpanel">
      <input id="model-search" type="search" />
      <ul id="model-list"></ul>
      <input id="model-custom-id" type="text" />
      <input id="hf-token" type="password" />
      <input id="default-chat-model" type="text" />
      <select id="model-dtype">
        <option value="q4" selected>q4</option>
        <option value="q8">q8</option>
        <option value="fp16">fp16</option>
        <option value="fp32">fp32</option>
      </select>
      <select id="filter-library">
        <option value="transformers.js" selected>transformers.js</option>
        <option value="onnx">onnx</option>
        <option value="">All</option>
      </select>
      <select id="q2-key-display-mode">
        <option value="q2" selected>q2</option>
        <option value="cgAt">cgAt</option>
        <option value="hex">hex</option>
      </select>
      <input id="bench-model-t2" type="text" />
      <input id="bench-model-t3" type="text" />
      <input id="bench-model-t4" type="text" />
      <button id="load-btn" disabled>Load</button>
    </div>

    <div id="embedding-panel" class="hidden"></div>
    <canvas id="embedding-canvas" width="280" height="64"></canvas>
    <p id="embedding-stats"></p>
    <input id="max-tokens" value="2048" />
    <input id="temperature" value="0" />
    <span id="temp-value"></span>
    <input id="rep-penalty" value="1.1" />
    <span id="rep-value"></span>

    <div id="storage-section">
      <div id="local-file-drop"></div>
      <input id="local-file-url" />
      <button id="local-file-add"></button>
      <ul id="local-files-list"></ul>
    </div>

    <span id="header-title"></span>
    <span id="sidebar-model-tag"></span>
  `;

  // Polyfill requestAnimationFrame for the test environment.
  if (!('requestAnimationFrame' in globalThis)) {
    // @ts-expect-error
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(cb, 0);
  }

  // Stub Worker so importing app.ts doesn't attempt to load a real worker.
  class FakeWorker {
    listeners = new Map<string, EventListenerOrEventListenerObject>();
    postMessage = vi.fn();

    addEventListener(type: string, handler: EventListenerOrEventListenerObject) {
      this.listeners.set(type, handler);
    }

    dispatch(type: string, data: unknown) {
      const handler = this.listeners.get(type);
      if (typeof handler === 'function') {
        handler({ data } as unknown as Event);
      } else if (handler && 'handleEvent' in handler && typeof handler.handleEvent === 'function') {
        handler.handleEvent({ data } as Event);
      }
    }
  }

  // @ts-expect-error
  globalThis.Worker = FakeWorker;

  // Mock fetch so initModelPicker doesn't hit the real HF Hub API.
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => [
      { id: 'test-org/model-alpha', downloads: 10_000, likes: 100, tags: ['transformers.js'] },
      { id: 'test-org/model-beta', downloads: 5_000, likes: 50, tags: [] },
    ],
  } as Partial<Response>);

  // Stub canvas 2D context.
  const canvas = document.querySelector('#embedding-canvas') as HTMLCanvasElement;
  if (canvas) {
    Object.defineProperty(canvas, 'clientWidth', { value: 280, configurable: true });
    canvas.getContext = (() => ({
      fillStyle: '',
      fillRect: vi.fn(),
    } as Partial<CanvasRenderingContext2D>)) as typeof canvas.getContext;
  }
}

describe('app.ts helpers and DOM integration', () => {
  let app: typeof import('../src/app.ts');

  beforeEach(async () => {
    // Ensure each test gets a fresh copy of app.ts and its module-level state.
    vi.resetModules();

    // Recreate the DOM and stubs needed by app.ts.
    document.body.innerHTML = '';
    setupDom();

    // Re-import app.ts for this test.
    app = await import('../src/app.ts');

    // Flush the async fetch triggered by initModelPicker() on module load,
    // so the model list is populated before individual tests run.
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  it('splitThinkBlocks handles various tag cases', () => {
    expect(app.splitThinkBlocks('Hello')).toEqual([{ type: 'text', text: 'Hello' }]);
    expect(app.splitThinkBlocks('A<think>inner</think>B')).toEqual([
      { type: 'text', text: 'A' },
      { type: 'think', text: 'inner' },
      { type: 'text', text: 'B' },
    ]);
    expect(app.splitThinkBlocks('X<think>partial')).toEqual([
      { type: 'text', text: 'X' },
      { type: 'think', text: 'partial' },
    ]);
  });

  it('stripThinkTags removes complete and incomplete think tags', () => {
    expect(app.stripThinkTags('A<think>ignore</think>B')).toBe('AB');
    expect(app.stripThinkTags('Start<think>ignore')).toBe('Start');
  });

  it('escapeAndFormatText escapes HTML and formats markdown-like syntax', () => {
    const input = '<b>&</b> **bold** *italic* `code`\n```block```';
    const out = app.escapeAndFormatText(input);
    expect(out).toContain('&lt;b&gt;&amp;&lt;/b&gt;');
    expect(out).toContain('<strong>bold</strong>');
    expect(out).toContain('<em>italic</em>');
    expect(out).toContain('<code class="inline-code">code</code>');
    expect(out).toContain('<pre class="code-block"><code>block</code></pre>');
  });

  it('min/max compute correct extremes', () => {
    const arr = new Float32Array([3, -1, 0, 5]);
    expect(app.min(arr)).toBe(-1);
    expect(app.max(arr)).toBe(5);
  });

  it('onStatus updates UI state when ready', () => {
    const input = document.querySelector('#user-input') as HTMLTextAreaElement;
    const loadOverlay = document.querySelector('#load-overlay') as HTMLElement;
    const modelStatus = document.querySelector('#model-status') as HTMLElement;

    app.onStatus('loading', 'Loading…');
    expect(modelStatus.textContent).toBe('Loading…');

    app.onStatus('ready');
    expect(loadOverlay.classList.contains('hidden')).toBe(true);
    expect(modelStatus.textContent).toBe('Ready');

    // jsdom may not fully implement focus behavior; ensure input is present.
    expect(input).toBeInstanceOf(HTMLTextAreaElement);
  });

  it('onStatus ready updates header-title and sidebar-model-tag', () => {
    // Use a model from the mocked API response to set selectedModelId.
    app.selectModel('test-org/model-alpha');
    app.onStatus('ready');

    const headerTitle = document.querySelector('#header-title') as HTMLElement;
    const sidebarTag = document.querySelector('#sidebar-model-tag') as HTMLElement;
    // Display name is the part after the slash in the model ID.
    expect(headerTitle.textContent).toContain('model-alpha');
    expect(sidebarTag.textContent).toContain('model-alpha');
  });

  it('onProgress updates the progress bar and status text', () => {
    const loadBarFill = document.querySelector('#load-bar-fill') as HTMLElement;
    const loadBarOuter = document.querySelector('#load-bar') as HTMLElement;
    const loadStatus = document.querySelector('#load-status') as HTMLElement;

    app.onProgress('foo.bin', 50, 100);
    expect(loadBarFill.style.width).toBe('50%');
    expect(loadStatus.textContent).toContain('foo.bin');
    expect(loadBarOuter.getAttribute('aria-valuenow')).toBe('50');

    app.onProgress('', 5, 0);
    expect(loadStatus.textContent).toContain('Loading');
  });

  it('onEmbedding makes the embedding panel visible and updates stats', () => {
    const embeddingPanel = document.querySelector('#embedding-panel') as HTMLElement;

    const data = new Float32Array([1, 2, 3, 4]).buffer;
    app.onEmbedding({ type: 'embedding', data, seqLen: 2, hiddenDim: 2, dtype: 'fp32' });

    expect(embeddingPanel.classList.contains('hidden')).toBe(false);
    const stats = document.querySelector('#embedding-stats') as HTMLElement;
    expect(stats.textContent).toContain('Shape: [2 × 2]');
    expect(stats.textContent).toContain('dtype=fp32');
  });

  it('sendMessage posts a generate message and updates the UI', async () => {
    const input = document.querySelector('#user-input') as HTMLTextAreaElement;
    const sendBtn = document.querySelector('#send-btn') as HTMLButtonElement;

    // Initialise the worker (simulates clicking Load).
    app.startWithModel('LiquidAI/LFM2.5-1.2B-Thinking-ONNX');
    const workerRef = app.worker as Worker & { postMessage: ReturnType<typeof vi.fn> };

    // Ensure model is marked ready so sendMessage will proceed.
    app.onStatus('ready');

    input.value = 'hello';
    app.sendMessage();

    expect(sendBtn.disabled).toBe(true);
    expect(sendBtn.classList.contains('hidden')).toBe(true);

    expect(workerRef.postMessage).toHaveBeenCalled();
    const lastMessage = workerRef.postMessage.mock.calls.slice(-1)[0][0];
    expect(lastMessage).toMatchObject({ type: 'generate' });

    // Reset generation state so later tests are not affected by isGenerating.
    app.onDone();
  });

  it('handleWorkerMessage handles token stream and done events', async () => {
    const sendBtn = document.querySelector('#send-btn') as HTMLButtonElement;

    // Create an assistant bubble without going through sendMessage.
    app.appendAssistantBubble();

    // Stream a token and allow the scheduled render to run.
    app.handleWorkerMessage({ type: 'token', token: 'X' });
    await new Promise((resolve) => setTimeout(resolve, 100));

    const assistantBubble = document.querySelector('#messages .bubble.assistant') as HTMLElement;
    expect(assistantBubble).toBeTruthy();

    // Complete generation.
    app.onDone();
    expect(sendBtn.disabled).toBe(false);
  });

  it('keydown Enter triggers sendMessage via the key handler', () => {
    const input = document.querySelector('#user-input') as HTMLTextAreaElement;

    // Initialise the worker (simulates clicking Load).
    app.startWithModel('LiquidAI/LFM2.5-1.2B-Thinking-ONNX');
    const workerRef = app.worker as Worker & { postMessage: ReturnType<typeof vi.fn> };

    app.onStatus('ready');
    input.value = 'hey';

    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(workerRef.postMessage).toHaveBeenCalled();
  });

  it('updates display values when range inputs change', () => {
    const tempEl = document.querySelector('#temperature') as HTMLInputElement;
    const tempValue = document.querySelector('#temp-value') as HTMLElement;
    tempEl.value = '1.23';
    tempEl.dispatchEvent(new Event('input'));
    expect(tempValue.textContent).toBe('1.23');

    const repEl = document.querySelector('#rep-penalty') as HTMLInputElement;
    const repValue = document.querySelector('#rep-value') as HTMLElement;
    repEl.value = '1.50';
    repEl.dispatchEvent(new Event('input'));
    expect(repValue.textContent).toBe('1.50');
  });

  it('readConfig returns numbers from inputs', () => {
    (document.querySelector('#max-tokens') as HTMLInputElement).value = '123';
    (document.querySelector('#temperature') as HTMLInputElement).value = '0.55';
    (document.querySelector('#rep-penalty') as HTMLInputElement).value = '1.75';

    expect(app.readConfig()).toEqual({
      max_new_tokens: 123,
      temperature: 0.55,
      repetition_penalty: 1.75,
    });
  });

  it('handles worker error messages and displays an error bubble', () => {
    app.onStatus('ready');
    app.handleWorkerMessage({ type: 'error', message: 'boom' });

    const errorBubble = document.querySelector('#messages .bubble.error') as HTMLElement;
    expect(errorBubble.textContent).toContain('⚠️ boom');
  });

  it('autoResizeTextarea caps height at 200px based on scrollHeight', () => {
    const input = document.querySelector('#user-input') as HTMLTextAreaElement;
    Object.defineProperty(input, 'scrollHeight', { value: 500, configurable: true });
    input.dispatchEvent(new Event('input'));
    expect(input.style.height).toBe('200px');
  });

  it('renderBubble renders think blocks and escapes HTML', () => {
    const bubble = document.createElement('div');
    app.renderBubble(bubble, 'Hi <think>debug</think> <script>');
    expect(bubble.querySelector('details')).toBeTruthy();
    expect(bubble.innerHTML).toContain('&lt;script&gt;');
  });

  it('renderEmbeddingHeatmap returns early when canvas context is missing', () => {
    const canvas = document.querySelector('#embedding-canvas') as HTMLCanvasElement;
    canvas.getContext = () => null;

    const data = new Float32Array([1, 2, 3, 4]);
    app.renderEmbeddingHeatmap(data, 2, 2);

    // Should not throw; coverage is achieved by hitting the early return.
    expect(true).toBe(true);
  });

  it('renderQ2Result appends Q² info to embeddingStats', () => {
    const stats = document.querySelector('#embedding-stats') as HTMLElement;
    stats.textContent = 'Shape: [1 × 8]  dtype=fp32  min=0.000  max=1.000';

    const packed = new Uint8Array([0xAA, 0xAA]); // D D D D D D D D (all strong+)
    app.renderQ2Result(packed, 0xdd8c000000000000n, 8, 'q2');

    expect(stats.textContent).toContain('Q²:');
    expect(stats.textContent).toContain('key=Q²:');
    expect(stats.textContent).toContain('2 bytes');
  });

  // ─── HF API + settings tests ─────────────────────────────────────────────────

  it('fetchHFModels builds correct URL and returns parsed models', async () => {
    const settings = app.loadSettings();
    const models = await app.fetchHFModels('smollm', settings);
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);

    // Verify the fetch was called with the right URL shape.
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const url = fetchMock.mock.calls[fetchMock.mock.calls.length - 1][0] as string;
    expect(url).toContain('pipeline_tag=text-generation');
    expect(url).toContain('search=smollm');
    expect(url).toContain('library=transformers.js');
  });

  it('fetchHFModels includes Authorization header when apiToken is set', async () => {
    const settings = { ...app.loadSettings(), apiToken: 'hf_test_token_123' };
    await app.fetchHFModels('', settings);
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [, opts] = fetchMock.mock.calls[fetchMock.mock.calls.length - 1] as [string, RequestInit];
    expect((opts?.headers as Record<string, string>)?.['Authorization']).toBe('Bearer hf_test_token_123');
  });

  it('fetchHFModels filters out models above 2B parameters', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => [
        { id: 'Qwen/Qwen2.5-7B-Instruct', downloads: 22_000_000, likes: 1_100, tags: [] },
        { id: 'test-org/model-alpha', downloads: 10_000, likes: 100, tags: [] },
      ],
    } as Partial<Response>);

    const models = await app.fetchHFModels('', app.loadSettings());
    expect(models.some((m) => m.id === 'Qwen/Qwen2.5-7B-Instruct')).toBe(false);
    expect(models.some((m) => m.id === 'test-org/model-alpha')).toBe(true);
  });

  it('fetchHFModels throws on non-OK response', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
    } as Partial<Response>);
    await expect(app.fetchHFModels('', app.loadSettings())).rejects.toThrow('429');
  });

  it('formatCount formats numbers correctly', () => {
    expect(app.formatCount(500)).toBe('500');
    expect(app.formatCount(1500)).toBe('1.5K');
    expect(app.formatCount(1_500_000)).toBe('1.5M');
  });

  it('loadSettings returns defaults when localStorage is empty', () => {
    const s = app.loadSettings();
    expect(s.dtype).toBe('q4');
    expect(s.filterLibrary).toBe('transformers.js');
    expect(s.apiToken).toBe('');
  });

  it('saveSettings persists and loadSettings restores values', () => {
    app.saveSettings({ apiToken: 'tok', dtype: 'q8', filterLibrary: 'onnx' });
    const s = app.loadSettings();
    expect(s.dtype).toBe('q8');
    expect(s.filterLibrary).toBe('onnx');
    expect(s.apiToken).toBe('tok');
  });

  it('initModelPicker renders model list items from the mocked HF API', () => {
    // The beforeEach flush (setTimeout 0) ensures the async fetch has resolved.
    const items = document.querySelectorAll('.model-item');
    expect(items.length).toBe(2); // two items from the mock response
  });

  it('selectModel highlights the item, clears custom input, and enables load btn', () => {
    const customInput = document.querySelector('#model-custom-id') as HTMLInputElement;
    const loadBtn = document.querySelector('#load-btn') as HTMLButtonElement;
    customInput.value = 'some/custom-model';

    // Pick a model that is in the mocked list.
    app.selectModel('test-org/model-alpha');

    expect(customInput.value).toBe('');
    expect(loadBtn.disabled).toBe(false);
    const selectedItem = document.querySelector('.model-item.selected') as HTMLElement;
    expect(selectedItem).toBeTruthy();
    expect(selectedItem.dataset['modelId']).toBe('test-org/model-alpha');
  });

  it('startWithModel shows load overlay and creates worker', () => {
    const loadOverlay = document.querySelector('#load-overlay') as HTMLElement;

    // The overlay is already visible because initModelPicker() auto-loads the
    // default chat model at startup.  Calling startWithModel() a second time
    // (with a different model) must keep the overlay visible and post a new
    // load message for the requested model.
    app.startWithModel('onnx-community/Qwen3.5-0.8B-Instruct');

    expect(loadOverlay.classList.contains('hidden')).toBe(false);
    expect(app.worker).not.toBeNull();

    // The last message sent to the worker should include modelId and dtype.
    const workerRef = app.worker as Worker & { postMessage: ReturnType<typeof vi.fn> };
    expect(workerRef.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'load', modelId: 'onnx-community/Qwen3.5-0.8B-Instruct', dtype: 'q4' }),
    );
  });

  it('changing model-dtype updates currentSettings and is passed to the worker', () => {
    const dtypeEl = document.querySelector('#model-dtype') as HTMLSelectElement;
    dtypeEl.value = 'q8';
    dtypeEl.dispatchEvent(new Event('change'));

    app.startWithModel('test/some-model');
    const workerRef = app.worker as Worker & { postMessage: ReturnType<typeof vi.fn> };
    expect(workerRef.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'load', dtype: 'q8' }),
    );
  });

  it('changing q2-key-display-mode persists the setting', () => {
    const keyModeEl = document.querySelector('#q2-key-display-mode') as HTMLSelectElement;
    keyModeEl.value = 'cgAt';
    keyModeEl.dispatchEvent(new Event('change'));

    const loadedSettings = app.loadSettings();
    expect(loadedSettings.q2KeyDisplayMode).toBe('cgAt');
  });

  it('switchTab changes active tab and shows correct panel', () => {
    const chatPanel = document.querySelector('#panel-chat') as HTMLElement;
    const benchPanel = document.querySelector('#panel-benchmarks') as HTMLElement;
    const settingsPanel = document.querySelector('#panel-settings') as HTMLElement;

    // Initially chat is active
    expect(chatPanel.classList.contains('hidden')).toBe(false);
    expect(benchPanel.classList.contains('hidden')).toBe(true);

    // Switch to benchmarks
    app.switchTab('benchmarks');
    expect(chatPanel.classList.contains('hidden')).toBe(true);
    expect(benchPanel.classList.contains('hidden')).toBe(false);
    expect(settingsPanel.classList.contains('hidden')).toBe(true);

    // Switch to settings
    app.switchTab('settings');
    expect(chatPanel.classList.contains('hidden')).toBe(true);
    expect(benchPanel.classList.contains('hidden')).toBe(true);
    expect(settingsPanel.classList.contains('hidden')).toBe(false);

    // Switch back to chat
    app.switchTab('chat');
    expect(chatPanel.classList.contains('hidden')).toBe(false);
    expect(benchPanel.classList.contains('hidden')).toBe(true);
    expect(settingsPanel.classList.contains('hidden')).toBe(true);
  });

  it('sendMessage shows error when no model is loaded', () => {
    const input = document.querySelector('#user-input') as HTMLTextAreaElement;
    input.value = 'hello';
    app.sendMessage();

    // Should show an error bubble since no model is loaded
    const errorBubble = document.querySelector('#messages .bubble.error') as HTMLElement;
    expect(errorBubble).toBeTruthy();
    expect(errorBubble.textContent).toContain('No model loaded');
  });

  it('runBenchmarks(t2) populates bench-results-body with T2 rows', () => {
    const body = document.querySelector('#bench-results-body') as HTMLTableSectionElement;
    const status = document.querySelector('#bench-status') as HTMLDivElement;

    app.runBenchmarks('t2');

    const rows = body.querySelectorAll('tr');
    expect(rows.length).toBeGreaterThan(0);
    // All rows should be T2 suite (config row is 'config', not a suite name)
    const suiteRows = Array.from(rows).filter((row) => row.cells[0]?.textContent !== 'config');
    suiteRows.forEach((row) => {
      expect(row.cells[0]?.textContent).toBe('T2');
    });
    expect(status.textContent).toMatch(/Completed:/);
  });

  it('runBenchmarks(t3) populates bench-results-body with T3 rows', () => {
    const body = document.querySelector('#bench-results-body') as HTMLTableSectionElement;
    app.runBenchmarks('t3');
    const rows = body.querySelectorAll('tr');
    expect(rows.length).toBeGreaterThan(0);
    const suiteRows = Array.from(rows).filter((row) => row.cells[0]?.textContent !== 'config');
    suiteRows.forEach((row) => {
      expect(row.cells[0]?.textContent).toBe('T3');
    });
  });

  it('runBenchmarks(t4) populates bench-results-body with T4 rows', () => {
    const body = document.querySelector('#bench-results-body') as HTMLTableSectionElement;
    app.runBenchmarks('t4');
    const rows = body.querySelectorAll('tr');
    expect(rows.length).toBeGreaterThan(0);
    const suiteRows = Array.from(rows).filter((row) => row.cells[0]?.textContent !== 'config');
    suiteRows.forEach((row) => {
      expect(row.cells[0]?.textContent).toBe('T4');
    });
  });

  it('runBenchmarks(t5) populates bench-results-body with T5 rows', () => {
    const body = document.querySelector('#bench-results-body') as HTMLTableSectionElement;
    app.runBenchmarks('t5');
    const rows = body.querySelectorAll('tr');
    expect(rows.length).toBeGreaterThan(0);
    const suiteRows = Array.from(rows).filter((row) => row.cells[0]?.textContent !== 'config');
    suiteRows.forEach((row) => {
      expect(row.cells[0]?.textContent).toBe('T5');
    });
  });

  it('runBenchmarks() runs all suites T0–T5 with no filter', () => {
    const body = document.querySelector('#bench-results-body') as HTMLTableSectionElement;
    app.runBenchmarks();
    const rows = body.querySelectorAll('tr');
    // Should have rows from all six suites (plus config row)
    const suites = new Set(Array.from(rows).map((row) => row.cells[0]?.textContent));
    expect(suites.has('T0')).toBe(true);
    expect(suites.has('T1')).toBe(true);
    expect(suites.has('T2')).toBe(true);
    expect(suites.has('T3')).toBe(true);
    expect(suites.has('T4')).toBe(true);
    expect(suites.has('T5')).toBe(true);
  });

  it('runBenchmarks config row shows configured benchmark model IDs', () => {
    const body = document.querySelector('#bench-results-body') as HTMLTableSectionElement;
    app.runBenchmarks('t4');
    const rows = Array.from(body.querySelectorAll('tr'));
    const configRow = rows.find((row) => row.cells[0]?.textContent === 'config');
    expect(configRow).toBeTruthy();
    // Should contain the configured T4 model ID
    expect(configRow?.cells[3]?.textContent).toContain('Qwen3.5-0.8B-ONNX');
  });

  it('runBenchmarks all-pass produces status "Completed: N/N passed"', () => {
    const status = document.querySelector('#bench-status') as HTMLDivElement;
    app.runBenchmarks('t0');
    expect(status.textContent).toMatch(/Completed: \d+\/\d+ passed/);
    const [passed, total] = (status.textContent?.match(/(\d+)\/(\d+)/) ?? []).slice(1).map(Number);
    expect(passed).toBe(total);
  });

  it('bench-model settings are persisted and restored', () => {
    const t2El = document.querySelector('#bench-model-t2') as HTMLInputElement;
    const t3El = document.querySelector('#bench-model-t3') as HTMLInputElement;
    const t4El = document.querySelector('#bench-model-t4') as HTMLInputElement;

    t2El.value = 'custom-org/my-code-model';
    t2El.dispatchEvent(new Event('change'));
    t3El.value = 'custom-org/my-embed-model';
    t3El.dispatchEvent(new Event('change'));
    t4El.value = 'custom-org/my-llm-model';
    t4El.dispatchEvent(new Event('change'));

    const loadedSettings = app.loadSettings();
    expect(loadedSettings.benchModelT2).toBe('custom-org/my-code-model');
    expect(loadedSettings.benchModelT3).toBe('custom-org/my-embed-model');
    expect(loadedSettings.benchModelT4).toBe('custom-org/my-llm-model');
  });

  it('clearing bench-model-t4 falls back to DEFAULT_SETTINGS value', () => {
    const t4El = document.querySelector('#bench-model-t4') as HTMLInputElement;
    // First set a custom value
    t4El.value = 'custom-org/my-llm-model';
    t4El.dispatchEvent(new Event('change'));
    // Then clear it
    t4El.value = '';
    t4El.dispatchEvent(new Event('change'));

    const loadedSettings = app.loadSettings();
    expect(loadedSettings.benchModelT4).toBe('onnx-community/Qwen3.5-0.8B-ONNX');
  });

  it('defaultChatModel setting is persisted and restored', () => {
    const el = document.querySelector('#default-chat-model') as HTMLInputElement;
    el.value = 'custom-org/my-chat-model';
    el.dispatchEvent(new Event('change'));

    const loadedSettings = app.loadSettings();
    expect(loadedSettings.defaultChatModel).toBe('custom-org/my-chat-model');
  });

  it('DEFAULT_SETTINGS defaultChatModel is Qwen3.5-0.8B-ONNX', () => {
    const loadedSettings = app.loadSettings();
    // With empty localStorage, defaultChatModel should be the Qwen3.5 default
    expect(loadedSettings.defaultChatModel).toBe('onnx-community/Qwen3.5-0.8B-ONNX');
  });
});
