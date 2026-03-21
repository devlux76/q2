/**
 * model-loading.spec.ts — Playwright E2E: verify that a real ONNX model can
 * be downloaded and loaded via transformers.js in a real Chromium browser.
 *
 * This is the acid test: NO mocks, NO fakes, NO fake DOM.  The transformers.js
 * pipeline downloads model weights from the Hugging Face Hub, initialises an
 * ONNX Runtime WASM session, and the UI transitions to "ready".
 *
 * The default model is onnx-community/Qwen3.5-0.8B-ONNX (q4).  Override with
 * the E2E_MODEL and E2E_DTYPE environment variables for faster CI runs.
 */
import { test, expect } from '@playwright/test';

/* Allow up to 5 minutes per test — model download + WASM init is slow. */
test.setTimeout(300_000);

const MODEL_ID = process.env.E2E_MODEL ?? 'onnx-community/Qwen3.5-0.8B-ONNX';
const MODEL_DTYPE = process.env.E2E_DTYPE ?? 'q4';

test.describe('Real model loading via transformers.js', () => {
  test('loads a model and transitions to ready state', async ({ page }, testInfo) => {
    // Collect console messages so we can diagnose failures.
    const logs: string[] = [];
    page.on('console', (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');

    // Navigate to Settings and enter the model ID.
    await page.click('#tab-settings');
    await page.fill('#model-custom-id', MODEL_ID);

    // Ensure the configured dtype is selected (q4 is the smallest download by default).
    await page.selectOption('#model-dtype', MODEL_DTYPE);

    await page.screenshot({ path: testInfo.outputPath('model-before-load.png'), fullPage: true });

    // Click Load — this triggers the real model download + WASM init.
    await page.click('#load-btn');

    // The loading overlay should appear.
    await expect(page.locator('#load-overlay')).toBeVisible({ timeout: 5_000 });
    await page.screenshot({ path: testInfo.outputPath('model-loading.png'), fullPage: true });

    // Wait for progress updates (download traffic).  The load-status element
    // should eventually show something other than "Initializing…".
    await page.waitForFunction(() => {
      const el = document.querySelector('#load-status');
      return el && el.textContent !== 'Initializing…';
    }, { timeout: 60_000 });

    await page.screenshot({ path: testInfo.outputPath('model-downloading.png'), fullPage: true });

    // Wait for the model status badge to show ready (model loaded successfully).
    // This is the big one — the full download + ONNX session init must succeed.
    await expect(page.locator('#model-status')).not.toContainText('No model', { timeout: 300_000 });

    // The loading overlay should disappear once the model is ready.
    await expect(page.locator('#load-overlay')).toBeHidden({ timeout: 300_000 });

    await page.screenshot({ path: testInfo.outputPath('model-loaded.png'), fullPage: true });

    // Dump console logs for debugging if something went wrong.
    if (logs.some((l) => l.includes('[error]'))) {
      console.log('--- Browser console logs ---');
      for (const l of logs) console.log(l);
      console.log('--- end ---');
    }
  });
});
