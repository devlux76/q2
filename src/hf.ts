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
  return (await res.json()) as HFModel[];
}
