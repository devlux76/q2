/**
 * chat.spec.ts — Playwright E2E: verify that a real chat round-trip works
 * end-to-end in a real browser with a real model.
 *
 * This test loads a model, sends a user message, and waits for the model to
 * generate a response.  No mocks, no fakes, no fake DOM.
 *
 * Timeout is generous because model loading + inference on WASM is slow.
 */
import { test, expect } from '@playwright/test';

/* Allow up to 8 minutes — model download + inference on CPU/WASM. */
test.setTimeout(480_000);

const MODEL_ID = process.env.E2E_MODEL ?? 'onnx-community/Qwen3.5-0.8B-ONNX';
const MODEL_DTYPE = process.env.E2E_DTYPE ?? 'q4';

test.describe('Real chat interaction', () => {
  test('sends a message and receives a streamed response', async ({ page }, testInfo) => {
    const logs: string[] = [];
    page.on('console', (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');

    // ── Step 1: Load the model ─────────────────────────────────────────────
    await page.click('#tab-settings');
    await page.fill('#model-custom-id', MODEL_ID);
    await page.selectOption('#model-dtype', MODEL_DTYPE);
    await page.click('#load-btn');

    // Wait for model to finish loading.
    await expect(page.locator('#load-overlay')).toBeHidden({ timeout: 300_000 });
    await expect(page.locator('#model-status')).not.toContainText('No model', { timeout: 5_000 });

    await page.screenshot({ path: testInfo.outputPath('chat-model-ready.png'), fullPage: true });

    // ── Step 2: Switch to Chat and send a message ──────────────────────────
    await page.click('#tab-chat');
    await expect(page.locator('#panel-chat')).toBeVisible();

    const inputEl = page.locator('#user-input');
    await inputEl.fill('Hello! What is 2+2?');
    await page.screenshot({ path: testInfo.outputPath('chat-message-typed.png'), fullPage: true });

    await page.click('#send-btn');

    // ── Step 3: Wait for the model to start generating ─────────────────────
    // The stop button appears while generating; send button is hidden.
    await expect(page.locator('#stop-btn')).toBeVisible({ timeout: 30_000 });

    await page.screenshot({ path: testInfo.outputPath('chat-generating.png'), fullPage: true });

    // ── Step 4: Wait for the response to complete ──────────────────────────
    // The send button reappears when generation is done.
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 180_000 });

    // There should be at least one assistant message in the chat log.
    const messages = page.locator('#messages');
    await expect(messages).not.toBeEmpty();

    // The message area should contain some non-trivial text from the model.
    const text = await messages.textContent();
    expect((text ?? '').length).toBeGreaterThan(5);

    await page.screenshot({ path: testInfo.outputPath('chat-response.png'), fullPage: true });

    // Dump logs if there were errors.
    if (logs.some((l) => l.includes('[error]'))) {
      console.log('--- Browser console logs ---');
      for (const l of logs) console.log(l);
      console.log('--- end ---');
    }
  });
});
