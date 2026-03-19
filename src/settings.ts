/**
 * settings.ts — Persistent application settings (localStorage)
 *
 * Settings are intentionally minimal; they are stored on the client and never
 * leave the user's device.
 */

import type { Dtype, FilterLibrary } from './types.js';

import type { Q2KeyDisplayMode } from './types.js';

export interface AppSettings {
  /** Optional HuggingFace API token (private models, higher rate limits). */
  apiToken: string;
  /** ONNX file suffix. */
  dtype: Dtype;
  /** Library filter tag sent to the HF Hub API. */
  filterLibrary: FilterLibrary;
  /** Q² transition key display mode in the embedding panel. */
  q2KeyDisplayMode: Q2KeyDisplayMode;
  /**
   * Default chat/inference model loaded when the app starts.
   * Users can override via Settings → Model or by selecting from the model list.
   */
  defaultChatModel: string;
  /**
   * Default model for T2 (structured code corpus) benchmarks.
   * Per TESTING.md §T2: sailesh27/unixcoder-base-onnx is the primary recommendation.
   * Users can override to compare models; leave blank to use the default.
   */
  benchModelT2: string;
  /**
   * Default model for T3 (Matryoshka / dedicated embedding) benchmarks.
   * Per TESTING.md §T3: Xenova/all-MiniLM-L6-v2 is the recommended tiny baseline.
   * Users can override to compare models; leave blank to use the default.
   */
  benchModelT3: string;
  /**
   * Default model for T4/T5 (standard local LLM / phylomemetic) benchmarks.
   * Per TESTING.md §T4: onnx-community/Qwen3.5-0.8B-ONNX is the recommended model.
   * Users can override to compare models; leave blank to use the default.
   */
  benchModelT4: string;
}

export const DEFAULT_SETTINGS: AppSettings = {
  apiToken: '',
  dtype: 'q4',
  filterLibrary: 'transformers.js',
  q2KeyDisplayMode: 'q2',
  defaultChatModel: 'onnx-community/Qwen3.5-0.8B-ONNX',
  benchModelT2: 'sailesh27/unixcoder-base-onnx',
  benchModelT3: 'Xenova/all-MiniLM-L6-v2',
  benchModelT4: 'onnx-community/Qwen3.5-0.8B-ONNX',
};

const SETTINGS_KEY = 'q2_settings';

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<AppSettings>) };
  } catch {
    // Ignore parse/storage errors
  }
  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(settings: AppSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage errors
  }
}
