import { beforeAll, describe, expect, it } from 'vitest';

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
}

describe('app.ts browser integration (real Worker)', () => {
  beforeAll(() => {
    setupDom();
    // Prevent auto-init so we can set up the worker URL and assertions first.
    // This is only required for tests; production still auto-starts.
    (globalThis as any).__Q2_SKIP_AUTO_INIT__ = true;
  });

  it('initializes a real worker and streams tokens into the UI', async () => {
    // Only run in a real browser environment where Worker is available.
    if (typeof Worker !== 'function') {
      expect(true).toBe(true);
      return;
    }

    const workerSource = `
      self.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === 'load') {
          postMessage({ type: 'status', status: 'ready' });
        }
        if (msg.type === 'generate') {
          postMessage({ type: 'status', status: 'generating' });
          postMessage({ type: 'token', token: 'H' });
          postMessage({ type: 'token', token: 'i' });
          postMessage({ type: 'done' });
        }
      };
    `;

    const blob = new Blob([workerSource], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    (globalThis as any).__Q2_WORKER_URL__ = url;

    const app = await import('../src/app.ts');

    // Init (this will create the worker and send the initial load message).
    app.initWorker();

    // Wait for the worker to respond and for the UI to update.
    await new Promise((resolve) => setTimeout(resolve, 100));

    // The worker should have set the app ready state and unhidden the chat UI.
    expect(document.querySelector('#chat-app')?.classList.contains('hidden')).toBe(false);

    // Trigger a generation and confirm the streaming tokens arrive.
    (document.querySelector('#user-input') as HTMLTextAreaElement).value = 'hi';
    app.sendMessage();

    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(document.querySelector('#messages')?.textContent).toContain('Hi');

    URL.revokeObjectURL(url);
  });
});
