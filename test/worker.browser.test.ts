import { describe, expect, it } from 'vitest';

// This test validates that a real Web Worker can be spawned in a true browser
// environment (e.g. via `bun run test:browser`). In jsdom/node mode, Web Workers
// are not available, so the test simply short-circuits.

describe('Web Worker integration (browser mode)', () => {
  it('can create a simple inline worker and exchange messages', async () => {
    if (typeof Worker !== 'function') {
      // Not running in a real browser (e.g. jsdom/node); skip the actual worker check.
      expect(true).toBe(true);
      return;
    }

    const code = `
      self.onmessage = (e) => {
        postMessage({ received: e.data });
      };
    `;

    const blob = new Blob([code], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url, { type: 'module' });

    const result = await new Promise<unknown>((resolve, reject) => {
      worker.onmessage = (event) => resolve(event.data);
      worker.onerror = (err) => reject(err);
      worker.postMessage('ping');
    });

    worker.terminate();
    expect(result).toEqual({ received: 'ping' });
  });
});
