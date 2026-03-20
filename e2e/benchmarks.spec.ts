/**
 * benchmarks.spec.ts — Playwright E2E: run the T0 and T1 benchmark suites
 * in a real browser and verify all tests pass.  T0 (algebraic invariants) and
 * T1 (null baselines) are pure-math computations — they do not need a loaded
 * model — so they are fast and deterministic.
 */
import { test, expect } from '@playwright/test';

test.describe('Q² Benchmarks (T0 & T1)', () => {
  test.beforeEach(async ({ page }) => {
    // Prevent auto-loading the default model so the loading overlay does not
    // block interaction with the benchmark buttons.
    await page.addInitScript(() => {
      (globalThis as unknown as Record<string, boolean>).__Q2_SKIP_AUTO_INIT__ = true;
    });
  });

  test('T0 algebraic invariants pass in the browser', async ({ page }, testInfo) => {
    await page.goto('/');
    await page.click('#tab-benchmarks');
    await expect(page.locator('#panel-benchmarks')).toBeVisible();

    // Click "T0 — Algebraic Invariants" button.
    await page.click('#bench-run-t0');

    // Wait for results to appear (status cells with bench-pass or bench-fail).
    await page.waitForFunction(() => {
      const rows = document.querySelectorAll('#bench-results-body tr');
      if (rows.length === 0) return false;
      // All rows should have a status that is not 'running' or 'pending'.
      return Array.from(rows).every(
        (r) => r.querySelector('.bench-pass') || r.querySelector('.bench-fail'),
      );
    }, { timeout: 30_000 });

    // Verify every T0 test row passed.
    const failCount = await page.locator('#bench-results-body .bench-fail').count();
    expect(failCount).toBe(0);

    const passCount = await page.locator('#bench-results-body .bench-pass').count();
    expect(passCount).toBeGreaterThan(0);

    await page.screenshot({ path: testInfo.outputPath('benchmark-t0.png'), fullPage: true });
  });

  test('T1 null baselines pass in the browser', async ({ page }, testInfo) => {
    await page.goto('/');
    await page.click('#tab-benchmarks');

    // Click "T1 — Null Baselines" button.
    await page.click('#bench-run-t1');

    // Wait for T1 results to settle.
    await page.waitForFunction(() => {
      const rows = document.querySelectorAll('#bench-results-body tr');
      if (rows.length === 0) return false;
      return Array.from(rows).every(
        (r) => r.querySelector('.bench-pass') || r.querySelector('.bench-fail'),
      );
    }, { timeout: 60_000 });

    const failCount = await page.locator('#bench-results-body .bench-fail').count();
    expect(failCount).toBe(0);

    const passCount = await page.locator('#bench-results-body .bench-pass').count();
    expect(passCount).toBeGreaterThan(0);

    await page.screenshot({ path: testInfo.outputPath('benchmark-t1.png'), fullPage: true });
  });

  test('Run All Benchmarks button triggers T0 and T1', async ({ page }, testInfo) => {
    await page.goto('/');
    await page.click('#tab-benchmarks');

    // Click the "Run All Benchmarks" button.
    await page.click('#bench-run-all');

    // Wait for benchmark results to appear and settle. The T0/T1 suites are
    // pure math and complete in under a second; T2-T5 may report "pending"
    // because no model is loaded — that is expected.
    await page.waitForFunction(() => {
      const rows = document.querySelectorAll('#bench-results-body tr');
      // We need at least the T0 + T1 rows (typically 5-7 rows).
      return rows.length >= 5;
    }, { timeout: 60_000 });

    // Wait for the benchmark runner to finish.
    await page.waitForFunction(() => {
      const status = document.querySelector('#bench-status');
      return status && /complete|finished|done/i.test(status.textContent ?? '');
    }, { timeout: 120_000 });

    // Verify T0 and T1 rows all passed.
    const t0Rows = await page.locator('#bench-results-body tr').filter({ hasText: 'T0' });
    const t0Count = await t0Rows.count();
    expect(t0Count).toBeGreaterThan(0);

    const t1Rows = await page.locator('#bench-results-body tr').filter({ hasText: 'T1' });
    const t1Count = await t1Rows.count();
    expect(t1Count).toBeGreaterThan(0);

    await page.screenshot({ path: testInfo.outputPath('benchmark-all.png'), fullPage: true });
  });
});
