import tsParser from '@typescript-eslint/parser';
import htmlParser from '@html-eslint/parser';
import htmlPlugin from '@html-eslint/eslint-plugin';
import ymlPlugin from 'eslint-plugin-yml';

export default [
  {
    ignores: ['dist/**', 'node_modules/**', 'coverage/**', 'gh-pages/**', 'e2e-results/**', 'test-results/**', 'playwright-report/**'],
  },
  {
    files: ['**/*.{js,ts}'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 2020,
        sourceType: 'module',
      },
    },
    plugins: {
      '@typescript-eslint': (await import('@typescript-eslint/eslint-plugin')).default,
    },
    rules: {
      'no-unused-vars': 'off',
      'no-undef': 'off',
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
    },
  },
  {
    files: ['**/*.html'],
    languageOptions: {
      parser: htmlParser,
    },
    plugins: {
      '@html-eslint': htmlPlugin,
    },
    rules: {},
  },
  ...ymlPlugin.configs['flat/recommended'],
];
