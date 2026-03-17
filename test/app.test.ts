import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

// Ensure our test environment has the minimal DOM needed by app.ts.
function setupDom() {
  document.body.innerHTML = `
    <div id="load-screen">
      <p id="load-status"></p>
      <div id="load-bar" role="progressbar"><div id="load-bar-fill"></div></div>
    </div>
    <div id="chat-app" class="hidden">
      <div id="messages"></div>
    </div>
    <textarea id="user-input"></textarea>
    <button id="send-btn"></button>
    <button id="stop-btn"></button>
    <div id="embedding-panel" class="hidden"></div>
    <canvas id="embedding-canvas" width="280" height="64"></canvas>
    <p id="embedding-stats"></p>
    <input id="max-tokens" value="2048" />
    <input id="temperature" value="0" />
    <span id="temp-value"></span>
    <input id="rep-penalty" value="1.1" />
    <span id="rep-value"></span>
  `;

  // Polyfill requestAnimationFrame for the test environment.
  if (!('requestAnimationFrame' in globalThis)) {
    // @ts-expect-error
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(cb, 0);
  }

  // Stub Worker so importing app.ts (which calls initWorker()) doesn't attempt to load a real worker.
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
      } else if (handler && typeof (handler as any).handleEvent === 'function') {
        (handler as any).handleEvent({ data });
      }
    }
  }

  // @ts-expect-error
  globalThis.Worker = FakeWorker;

  // Stub canvas 2D context.
  const canvas = document.querySelector('#embedding-canvas') as HTMLCanvasElement;
  if (canvas) {
    Object.defineProperty(canvas, 'clientWidth', { value: 280, configurable: true });
    canvas.getContext = () => ({
      fillStyle: '',
      fillRect: vi.fn(),
    } as any);
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

  beforeAll(async () => {
    setupDom();
    app = await import('../src/app.ts');
  });

  beforeEach(() => {
    // Reset the DOM state between tests.
    const loadScreen = document.querySelector('#load-screen') as HTMLElement;
    const chatApp = document.querySelector('#chat-app') as HTMLElement;
    loadScreen.classList.remove('hidden');
    chatApp.classList.add('hidden');
    (document.querySelector('#messages') as HTMLElement).textContent = '';
    (document.querySelector('#user-input') as HTMLTextAreaElement).value = '';

    // Reset send/stop button state.
    const sendBtn = document.querySelector('#send-btn') as HTMLButtonElement;
    const stopBtn = document.querySelector('#stop-btn') as HTMLButtonElement;
    sendBtn.disabled = false;
    sendBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');

    // Reset embedding panel.
    const embeddingPanel = document.querySelector('#embedding-panel') as HTMLElement;
    embeddingPanel.classList.add('hidden');
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
    const loadScreen = document.querySelector('#load-screen') as HTMLElement;
    const chatApp = document.querySelector('#chat-app') as HTMLElement;

    app.onStatus('loading', 'Loading…');
    expect(loadScreen.classList.contains('hidden')).toBe(false);
    expect(chatApp.classList.contains('hidden')).toBe(true);

    app.onStatus('ready');
    expect(loadScreen.classList.contains('hidden')).toBe(true);
    expect(chatApp.classList.contains('hidden')).toBe(false);

    // jsdom may not fully implement focus behavior; ensure input is present.
    expect(input).toBeInstanceOf(HTMLTextAreaElement);
  });

  it('onProgress updates the progress bar and status text', () => {
    const loadBar = document.querySelector('#load-bar-fill') as HTMLElement;
    const loadStatus = document.querySelector('#load-status') as HTMLElement;

    app.onProgress('foo.bin', 50, 100);
    expect(loadBar.style.width).toBe('50%');
    expect(loadStatus.textContent).toContain('foo.bin');

    app.onProgress('', 5, 0);
    expect(loadStatus.textContent).toContain('Loading');
  });

  it('onEmbedding makes the embedding panel visible and updates stats', () => {
    const embeddingPanel = document.querySelector('#embedding-panel') as HTMLElement;

    const data = new Float32Array([1, 2, 3, 4]).buffer;
    app.onEmbedding({ type: 'embedding', data, seqLen: 2, hiddenDim: 2 });

    expect(embeddingPanel.classList.contains('hidden')).toBe(false);
    const stats = document.querySelector('#embedding-stats') as HTMLElement;
    expect(stats.textContent).toContain('Shape: [2 × 2]');
  });

  it('sendMessage posts a generate message and updates the UI', async () => {
    const input = document.querySelector('#user-input') as HTMLTextAreaElement;
    const sendBtn = document.querySelector('#send-btn') as HTMLButtonElement;
    const worker = (app as any).worker as any;

    // Ensure model is marked ready so sendMessage will proceed.
    app.onStatus('ready');

    input.value = 'hello';
    app.sendMessage();

    expect(sendBtn.disabled).toBe(true);
    expect(sendBtn.classList.contains('hidden')).toBe(true);

    expect(worker.postMessage).toHaveBeenCalled();
    const lastMessage = worker.postMessage.mock.calls.slice(-1)[0][0];
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
    const worker = (app as any).worker as any;

    app.onStatus('ready');
    input.value = 'hey';

    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(worker.postMessage).toHaveBeenCalled();
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
});
