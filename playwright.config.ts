import { defineConfig } from '@playwright/test';

/**
 * Playwright E2E configuration for Q².
 *
 * Serves the built application via a lightweight static server and runs real
 * browser tests against it — no mocks, no fakes, no fake DOM.
 */

/**
 * Chromium launch args for WebGPU support.
 *
 * Local Docker runs (e2e/run-local.sh):
 *   /dev/dri is passed to the container for Intel/AMD GPU access.  The run
 *   script sets E2E_GPU_AVAILABLE=1 automatically when /dev/dri is present,
 *   which drops --use-gl=swiftshader and lets Chromium use hardware Vulkan
 *   (Intel ANV / AMD RADV).
 *
 * Without GPU (lavapipe fallback):
 *   --use-gl=swiftshader forces the Mesa software Vulkan rasterizer.
 *   WebGPU still works; inference falls back to WASM.
 */
const gpuArgs = [
  '--enable-gpu',
  '--ignore-gpu-blocklist',
  '--enable-unsafe-webgpu',
  '--disable-gpu-sandbox',
  // Use SwiftShader (software GL) when no GPU is available.
  // Dropped when E2E_GPU_AVAILABLE=1 (set by run-local.sh on /dev/dri systems).
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
     * On systems with Intel/AMD GPU: hardware Vulkan via /dev/dri passthrough.
     * Without GPU: Mesa lavapipe (software Vulkan) via SwiftShader flag. */
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
