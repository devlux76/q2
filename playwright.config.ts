import { defineConfig } from '@playwright/test';

/**
 * Playwright E2E configuration for Q².
 *
 * Serves the built application via a lightweight static server and runs real
 * browser tests against it — no mocks, no fakes, no fake DOM.
 */

/**
 * Chromium launch args for WebGPU support in CI.
 *
 * On CPU-only runners (ubuntu-latest, no GPU hardware):
 *   --use-gl=swiftshader forces Chromium to use the Mesa/SwiftShader software
 *   rasterizer.  WebGPU then runs through the Vulkan-over-SwiftShader adapter,
 *   which is slow but always available.  WASM is the final inference fallback.
 *
 * On GPU runners (ubuntu-latest-gpu / self-hosted with NVIDIA):
 *   Drop --use-gl=swiftshader so Chromium can use the real hardware GPU via
 *   Vulkan/ANGLE.  Set the E2E_GPU_AVAILABLE=1 repository variable (see
 *   ci.yml comments) to activate this path.
 */
const gpuArgs = [
  '--enable-gpu',
  '--ignore-gpu-blocklist',
  '--enable-unsafe-webgpu',
  '--disable-gpu-sandbox',
  // Use SwiftShader (software GL) only on CPU-only runners.  On a real GPU
  // runner set E2E_GPU_AVAILABLE=1 to let Chrome use hardware rendering.
  ...( process.env.E2E_GPU_AVAILABLE ? [] : ['--use-gl=swiftshader'] ),
];

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
     * On CPU-only runners SwiftShader software-GL is used instead
     * (see gpuArgs above). */
    launchOptions: {
      args: gpuArgs,
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
