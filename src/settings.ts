/**
 * settings.ts — Persistent application settings (localStorage)
 *
 * Settings are intentionally minimal; they are stored on the client and never
 * leave the user's device.
 */

export interface AppSettings {
  /** Optional HuggingFace API token (private models, higher rate limits). */
  apiToken: string;
  /** ONNX file suffix: 'q4' | 'q8' | 'fp16' | 'fp32'. */
  dtype: string;
  /**
   * library filter sent to the HF Hub API.
   * 'transformers.js' — models tagged for the transformers.js runtime (default)
   * 'onnx'            — all ONNX models
   * ''                — no filter (all text-generation models)
   */
  filterLibrary: string;
}

export const DEFAULT_SETTINGS: AppSettings = {
  apiToken: '',
  dtype: 'q4',
  filterLibrary: 'transformers.js',
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
