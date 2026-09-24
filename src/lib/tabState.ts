// Which top-level tab (Calculator, Planner) is active. Framework-free URL codec only; the
// tab bar itself lives in src/ui/TabBar.tsx.

export const TAB_KEYS = ['calculator', 'planner'] as const;
export type TabKey = (typeof TAB_KEYS)[number];

const PARAM = 'tab';
const DEFAULT_TAB: TabKey = 'calculator';

/**
 * Decodes the active tab from a query string (with or without the leading "?"). A missing or
 * unrecognized value is Calculator, so every link written before tabs existed — and every
 * `c2`/`c3` compare link — opens exactly as it did before.
 */
export function decodeTab(search: string): TabKey {
  const qs = search.startsWith('?') ? search.slice(1) : search;
  const raw = new URLSearchParams(qs).get(PARAM);
  return (TAB_KEYS as readonly string[]).includes(raw ?? '') ? (raw as TabKey) : DEFAULT_TAB;
}

/**
 * Sets (or removes) the `tab` key on an existing URLSearchParams in place. Calculator is the
 * implicit default, so it is never written — an old link and a fresh Calculator link stay
 * byte-for-byte identical, and only Planner (and any future non-default tab) grows the URL.
 */
export function setTabParam(params: URLSearchParams, tab: TabKey): void {
  if (tab === DEFAULT_TAB) params.delete(PARAM);
  else params.set(PARAM, tab);
}
