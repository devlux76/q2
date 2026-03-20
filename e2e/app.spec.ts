/**
 * app.spec.ts — Playwright E2E: verify the Q² application loads and the
 * basic UI (tabs, settings panel, benchmark panel) is functional in a real
 * browser with no mocks or fakes.
 */
import { test, expect } from '@playwright/test';

test.describe('Q² Application UI', () => {
  test.beforeEach(async ({ page }) => {
    // Prevent auto-loading the default model so the loading overlay does not
    // block interaction with UI elements.
    await page.addInitScript(() => {
      (globalThis as unknown as Record<string, boolean>).__Q2_SKIP_AUTO_INIT__ = true;
    });
  });

  test('loads the app and displays the navigation bar', async ({ page }, testInfo) => {
    await page.goto('/');
    await expect(page.locator('#top-nav')).toBeVisible();
    await expect(page.locator('.nav-logo')).toHaveText('Q²');
    await expect(page.locator('#tab-chat')).toBeVisible();
    await expect(page.locator('#tab-benchmarks')).toBeVisible();
    await expect(page.locator('#tab-settings')).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('app-loaded.png'), fullPage: true });
  });

  test('navigates between tabs', async ({ page }, testInfo) => {
    await page.goto('/');

    // Chat tab should be active by default.
    await expect(page.locator('#panel-chat')).toBeVisible();
    await expect(page.locator('#panel-benchmarks')).toBeHidden();
    await expect(page.locator('#panel-settings')).toBeHidden();

    // Switch to Benchmarks tab.
    await page.click('#tab-benchmarks');
    await expect(page.locator('#panel-benchmarks')).toBeVisible();
    await expect(page.locator('#panel-chat')).toBeHidden();
    await page.screenshot({ path: testInfo.outputPath('benchmarks-tab.png'), fullPage: true });

    // Switch to Settings tab.
    await page.click('#tab-settings');
    await expect(page.locator('#panel-settings')).toBeVisible();
    await expect(page.locator('#panel-benchmarks')).toBeHidden();
    await page.screenshot({ path: testInfo.outputPath('settings-tab.png'), fullPage: true });

    // Return to Chat.
    await page.click('#tab-chat');
    await expect(page.locator('#panel-chat')).toBeVisible();
  });

  test('settings panel has model configuration fields', async ({ page }, testInfo) => {
    await page.goto('/');
    await page.click('#tab-settings');

    await expect(page.locator('#default-chat-model')).toBeVisible();
    await expect(page.locator('#model-dtype')).toBeVisible();
    await expect(page.locator('#filter-library')).toBeVisible();
    await expect(page.locator('#load-btn')).toBeVisible();
    await expect(page.locator('#model-custom-id')).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('settings-panel.png'), fullPage: true });
  });

  test('model status badge starts with "No model"', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#model-status')).toContainText('No model');
  });
});
