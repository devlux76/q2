import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
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
