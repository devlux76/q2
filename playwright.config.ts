import { defineConfig } from '@playwright/test';

/**
 * Playwright E2E configuration for Q².
 *
 * Serves the built application via a lightweight static server and runs real
 * browser tests against it — no mocks, no fakes, no fake DOM.
 */
export default defineConfig({
  testDir: './e2e',
  outputDir: './e2e-results',

  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: Boolean(process.env.CI),

  /* Retry once on CI to absorb network flakiness during model download. */
  retries: process.env.CI ? 1 : 0,

  /* Run tests sequentially — model loading is heavy. */
  workers: 1,

  reporter: process.env.CI ? 'github' : 'list',

  use: {
    baseURL: 'http://localhost:4173',
    /* Take a screenshot on every test so the issue reporter can see proof. */
    screenshot: 'on',
    trace: 'retain-on-failure',
    headless: true,
    /* Enable WebGPU and hardware-acceleration hints.
     * On runners with a real GPU these flags allow WebGPU inference.
     * On CPU-only runners they engage SwiftShader software-GL so the
     * WebGPU preflight succeeds and WASM is used as final fallback. */
    launchOptions: {
      args: [
        '--enable-gpu',
        '--ignore-gpu-blocklist',
        '--enable-unsafe-webgpu',
        '--disable-gpu-sandbox',
        '--use-gl=swiftshader',
      ],
    },
  },

  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],

  /* Start a static file server before running tests. */
  webServer: {
    command: 'node e2e/serve.mjs',
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 10_000,
  },
});
