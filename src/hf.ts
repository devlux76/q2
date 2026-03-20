/**
 * hf.ts — HuggingFace Hub API helpers
 *
 * Provides model discovery and metadata formatting for the HF Hub API.
 * No API key is required for public models; providing one enables private
 * models and increases the rate limit.
 */

import type { AppSettings } from './settings.js';

/** Model record returned by the HF Hub API. */
export interface HFModel {
  id: string;
  downloads: number;
  likes: number;
  tags: string[];
}

/** Format a large number as a compact string (e.g. 1234567 → "1.2M"). */
export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/**
 * Estimate parameter count from a model ID string (e.g. "foo-1.5B" => 1.5e9).
 * Returns null when parameter count cannot be determined.
 */
export function parseModelParameterCount(modelId: string): number | null {
  const re = /([0-9]+(?:\.[0-9]+)?)([kKmMbB])/g;
  let match: RegExpExecArray | null;
  let maxCount: number | null = null;

  while ((match = re.exec(modelId)) !== null) {
    const value = Number(match[1]);
    const unitRaw = match[2];
    if (!unitRaw || !Number.isFinite(value)) continue;

    const unit = unitRaw.toLowerCase();
    let count: number;
    if (unit === 'k') count = value * 1e3;
    else if (unit === 'm') count = value * 1e6;
    else if (unit === 'b') count = value * 1e9;
    else continue;

    if (maxCount === null || count > maxCount) {
      maxCount = count;
    }
  }

  return maxCount;
}

const MAX_HF_MODEL_PARAMS = 2e9;

/**
 * Keep models of known or inferred size <= 2 billion parameters.
 */
export function isModelWithinParameterLimit(model: HFModel): boolean {
  const count = parseModelParameterCount(model.id);
  return count === null || count <= MAX_HF_MODEL_PARAMS;
}

/**
 * Fetch text-generation models from the HuggingFace Hub API.
 */
export async function fetchHFModels(query: string, settings: AppSettings): Promise<HFModel[]> {
  const params = new URLSearchParams({
    pipeline_tag: 'text-generation',
    sort: 'downloads',
    direction: '-1',
    limit: '20',
  });
  if (query.trim()) params.set('search', query.trim());
  if (settings.filterLibrary) params.set('library', settings.filterLibrary);

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (settings.apiToken) headers['Authorization'] = `Bearer ${settings.apiToken}`;

  const res = await fetch(`https://huggingface.co/api/models?${params}`, { headers });
  if (!res.ok) throw new Error(`HF API ${res.status}: ${res.statusText}`);

  const allModels = (await res.json()) as HFModel[];
  const filtered = allModels.filter(isModelWithinParameterLimit);
  return filtered;
}
