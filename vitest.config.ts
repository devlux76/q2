import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    exclude: ['node_modules', 'dist', 'e2e'],
    browser: {
      // Disabled by default; enable using `--browser`.
      enabled: false,
      provider: playwright(),
      instances: [{ browser: 'chromium' }],
    },
    coverage: {
      provider: 'istanbul',
      reporter: ['text', 'lcov'],
      reportsDirectory: 'coverage',
      // Require at least 80% overall coverage.
      lines: 80,
      functions: 80,
      branches: 80,
      statements: 80,
    },
  },
});
