// Compare mode: 2-3 CalcState columns side by side. Framework-free (rendering lives in
// src/ui/CompareTable.tsx); this module only knows how to encode/decode extra columns into
// the URL and how to rank a row of computed results.

import { decodeState, encodeState } from './urlState';
import type { CalcResult, CalcState } from './types';

/** The primary column plus up to this many extra columns (2 total minimum for "compare"). */
export const MAX_COMPARE_COLUMNS = 3;

/** Compact URL key for extra column n (2 or 3). */
function columnKey(n: number): string {
  return `c${n}`;
}

/**
 * Encodes the primary column plus any extra compare columns into one query string. Column 1
 * (primary) keeps today's flat keys via the existing `encodeState`, so a single-config link
 * (no extra columns) is byte-for-byte identical to before compare mode existed. Each extra
 * column is packed under its own key (`c2`, `c3`) as a *whole* `encodeState` query string, so
 * any CalcState field either this branch or another one adds later round-trips automatically —
 * nothing here lists individual fields.
 */
export function encodeCompareState(columns: readonly CalcState[]): string {
  const params = new URLSearchParams(encodeState(columns[0]));
  for (let i = 1; i < columns.length && i < MAX_COMPARE_COLUMNS; i++) {
    params.set(columnKey(i + 1), encodeState(columns[i]));
  }
  return params.toString();
}

/**
 * Decodes the extra compare columns (2 and 3) out of the URL. The primary column is decoded
 * separately by the existing `initialState`/`decodeState` path and is untouched by this
 * function. Stops at the first missing or invalid column so columns stay contiguous (never A/_/C).
 */
export function decodeCompareColumns(search: string): CalcState[] {
  const qs = search.startsWith('?') ? search.slice(1) : search;
  const q = new URLSearchParams(qs);
  const columns: CalcState[] = [];
  for (let n = 2; n <= MAX_COMPARE_COLUMNS; n++) {
    const raw = q.get(columnKey(n));
    if (raw === null) break;
    const failure = {} as CalcState; // unique per call; decodeState hands this back unchanged on failure
    const decoded = decodeState(raw, failure);
    if ((decoded as unknown) === failure) break;
    columns.push(decoded);
  }
  return columns;
}

/** A/B/C header labels for up to MAX_COMPARE_COLUMNS columns. */
export function columnLabel(index: number): string {
  return String.fromCharCode('A'.charCodeAt(0) + index);
}

export interface CompareRowSpec {
  key: string;
  label: string;
  unit: 'bytes' | 'count' | 'tokens' | 'tokS';
  higherIsBetter: boolean;
  /**
   * True for a row whose number only means anything once the config actually loads (e.g.
   * decode tok/s, or the max users/context that assumes the model is resident). A column that
   * doesn't fit is excluded from winning that row — the figure is shown, just never highlighted
   * as "better", since it isn't actually achievable on that column's hardware.
   */
  excludeUnfitFromBest?: boolean;
  value: (r: CalcResult) => number;
}

/** Headline figures shown in the comparison table, in display order. */
export const COMPARE_ROWS: readonly CompareRowSpec[] = [
  { key: 'weights', label: 'Weights', unit: 'bytes', higherIsBetter: false, value: (r) => r.weightBytes },
  { key: 'kvPerRequest', label: 'KV per request', unit: 'bytes', higherIsBetter: false, value: (r) => r.kvBytesPerRequest },
  { key: 'totalVram', label: 'Total VRAM', unit: 'bytes', higherIsBetter: false, value: (r) => r.totalBytes },
  { key: 'headroom', label: 'Headroom', unit: 'bytes', higherIsBetter: true, value: (r) => r.headroomBytes },
  { key: 'maxUsers', label: 'Max users', unit: 'count', higherIsBetter: true, excludeUnfitFromBest: true, value: (r) => r.maxUsersAtContext },
  { key: 'maxContext', label: 'Max context', unit: 'tokens', higherIsBetter: true, excludeUnfitFromBest: true, value: (r) => r.maxContextForUsers },
  { key: 'tokS', label: 'Tok/s (aggregate)', unit: 'tokS', higherIsBetter: true, excludeUnfitFromBest: true, value: (r) => r.throughput.aggregateTokS },
];

/**
 * Index of the best value in a row, or -1 when the best is tied (nothing is highlighted, per
 * spec), every value is NaN, or `eligible` (when given) rules every value out. `Infinity` is a
 * legitimate value (e.g. "unlimited" max users) and always wins/loses as the extreme it is, so
 * only NaN is excluded from comparison on that basis. `eligible[i] === false` excludes column i
 * from winning regardless of its value (e.g. a column that doesn't fit; see
 * `CompareRowSpec.excludeUnfitFromBest`) — the value is still shown, just never highlighted.
 */
export function bestColumnIndex(values: readonly number[], higherIsBetter: boolean, eligible?: readonly boolean[]): number {
  const candidates = values.map((v, i) => ({ v, i })).filter((c) => !Number.isNaN(c.v) && (eligible ? eligible[c.i] : true));
  if (candidates.length === 0) return -1;
  const best = higherIsBetter ? Math.max(...candidates.map((c) => c.v)) : Math.min(...candidates.map((c) => c.v));
  const winners = candidates.filter((c) => c.v === best);
  if (winners.length !== 1) return -1;
  return winners[0].i;
}

/** Deep-clones a CalcState for a new compare column (plain JSON data, no functions/dates). */
export function cloneState(state: CalcState): CalcState {
  return JSON.parse(JSON.stringify(state)) as CalcState;
}
