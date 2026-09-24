// Hub search-as-you-type: fetches id/downloads/gated for a query, and fuzzy-matches
// the bundled presets so the results list still works offline (see hfSearch.test.ts).

export interface HubSearchHit {
  id: string;
  downloads: number;
  gated: boolean;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function toHit(v: unknown): HubSearchHit | undefined {
  if (!isObject(v) || typeof v.id !== 'string') return undefined;
  const downloads = typeof v.downloads === 'number' && Number.isFinite(v.downloads) ? v.downloads : 0;
  // The API returns `gated: false` when not gated, or a string ("manual" | "auto") when it is.
  const gated = v.gated !== false && v.gated != null;
  return { id: v.id, downloads, gated };
}

/**
 * Search huggingface.co/api/models by repo id substring, sorted by downloads.
 * Never throws: a network error, a non-OK response, or unparsable JSON all resolve to [].
 */
export async function searchHub(
  query: string,
  token?: string,
  fetchImpl: typeof fetch = fetch,
  limit = 10,
): Promise<HubSearchHit[]> {
  const q = query.trim();
  if (!q) return [];
  const headers: Record<string, string> = {};
  if (token && token.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const url = `https://huggingface.co/api/models?search=${encodeURIComponent(q)}&limit=${limit}&sort=downloads&expand[]=gated`;
  try {
    const res = await fetchImpl(url, { headers });
    if (!res.ok) return [];
    const json: unknown = await res.json();
    if (!Array.isArray(json)) return [];
    const hits: HubSearchHit[] = [];
    for (const item of json) {
      const hit = toHit(item);
      if (hit) hits.push(hit);
    }
    return hits;
  } catch {
    return [];
  }
}

/** Case-insensitive subsequence test: every character of `query` appears in `target`, in order. */
function isSubsequence(query: string, target: string): boolean {
  let i = 0;
  for (let j = 0; j < target.length && i < query.length; j++) {
    if (target[j] === query[i]) i++;
  }
  return i === query.length;
}

/**
 * Fuzzy-match a query against a repo id or display name: a substring match ranks first,
 * then an in-order subsequence match (so "lla3" finds "meta-llama/Llama-3.1-8B-Instruct").
 */
export function fuzzyMatches(query: string, id: string, name: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return false;
  const idL = id.toLowerCase();
  const nameL = name.toLowerCase();
  return idL.includes(q) || nameL.includes(q) || isSubsequence(q, idL) || isSubsequence(q, nameL);
}
