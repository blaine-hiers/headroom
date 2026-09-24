// Planner tab state. Deliberately separate from the Calculator's reducer (src/ui/state.ts) so
// switching tabs never loses either side's work: Calculator <-> Planner <-> Calculator round
// trips both untouched.
//
// Contract for #24 (TaskPicker) and #25 (HardwareSizing): each step gets its own optional
// sub-state slice here (`taskPicker` / `hardwareSizing`) plus its own action(s) below, instead of
// widening a shared shape. `handoffModelId` is the one field both steps already share — step 1
// ("Which model for this task?") sets it via `setHandoffModelId`, step 2 ("What hardware to
// serve N users?") reads it to preselect the model it sizes hardware for.

import {
  DEFAULT_TASK_PICKER_CONSTRAINTS,
  findModelPreset,
  GPU_VENDOR_GROUPS,
  KV_QUANTS,
  MAX_CATALOG_CONTEXT,
  MAX_USERS,
  MIN_CONTEXT,
  RUNTIME_KEYS,
  WEIGHT_QUANTS,
} from '../../lib';
import type { GpuVendor, HardwareSizingSort, KvQuantKey, RuntimeKey, TaskPickerConstraints, WeightQuantKey } from '../../lib';

const HARDWARE_SIZING_SORTS: readonly HardwareSizingSort[] = ['smallest', 'cheapest'];

// Sane upper bounds for fields the HardwareSizing UI itself doesn't hard-cap the same way the
// Calculator's own inputs do — a crafted `phmt`/`phtt` must still be clamped to something a real
// user could have typed there (matches HardwareSizing.tsx's own NumberField max props).
const MAX_MIN_PER_USER_TOK_S = 1000;
const MAX_TTFT_SECONDS = 60;

/** Step 2's own inputs (issue #25). Kept as one sub-object so it can be reset/patched as a unit. */
export interface HardwareSizingState {
  /** Explicit model pick (a catalog id), when the user overrides the handoff/current-Calculator model. */
  modelId?: string;
  /** Size for the Calculator's current primary-column model instead of modelId/handoffModelId. */
  useCalculatorModel: boolean;
  concurrentUsers: number;
  contextTokens: number;
  weightQuant: WeightQuantKey;
  kvQuant: KvQuantKey;
  runtime: RuntimeKey;
  /** Minimum acceptable per-user decode tok/s. */
  minPerUserTokS: number;
  /** Maximum acceptable time-to-first-token, seconds. Undefined = not checked. */
  maxTtftSeconds?: number;
  /** Restrict the search to one vendor group; undefined = all vendors. */
  vendor?: GpuVendor;
  offloadEnabled: boolean;
  /** How to order qualifying rows; see HardwareSizingSort. */
  sort: HardwareSizingSort;
}

export const DEFAULT_HARDWARE_SIZING: HardwareSizingState = {
  useCalculatorModel: false,
  concurrentUsers: 32,
  contextTokens: 8192,
  weightQuant: 'q4_k_m',
  kvQuant: 'fp16',
  runtime: 'generic',
  minPerUserTokS: 20,
  offloadEnabled: false,
  sort: 'smallest',
};

export interface PlannerState {
  /** Set by TaskPicker (#24) once it recommends a model; read by HardwareSizing (#25). */
  handoffModelId?: string;
  /** The TaskPicker's own inputs (#24); absent until the user changes one, so a fresh Planner visit stays out of the URL. */
  taskPicker?: TaskPickerConstraints;
  /** Step 2's inputs (#25); absent until the user first touches HardwareSizing. */
  hardwareSizing?: HardwareSizingState;
}

export const initialPlannerState: PlannerState = {};

export type PlannerAction =
  | { type: 'setHandoffModelId'; modelId: string | undefined }
  | { type: 'taskPicker/setConstraints'; patch: Partial<TaskPickerConstraints> }
  | { type: 'hardwareSizing/patch'; patch: Partial<HardwareSizingState> }
  | { type: 'clear' }
  | { type: 'restore'; state: PlannerState };

export function plannerReducer(state: PlannerState, action: PlannerAction): PlannerState {
  switch (action.type) {
    case 'setHandoffModelId': {
      if (action.modelId === undefined) {
        const { handoffModelId: _drop, ...rest } = state;
        return rest;
      }
      // The handoff must win over whatever step 2 already had explicitly picked, or "Size
      // hardware" silently does nothing (#27 review item 3) — so this also clears
      // hardwareSizing's own modelId/useCalculatorModel by writing the handed-off id straight
      // into hardwareSizing.modelId instead. That keeps one source of truth and means the
      // existing `phm` URL key already persists the handoff across reload/share (#27 review item
      // 4), with no separate `handoffModelId` key needed. HardwareSizing.tsx's resolveModel still
      // labels it "from step 1" by comparing hs.modelId back against handoffModelId.
      return {
        ...state,
        handoffModelId: action.modelId,
        hardwareSizing: { ...(state.hardwareSizing ?? DEFAULT_HARDWARE_SIZING), modelId: action.modelId, useCalculatorModel: false },
      };
    }
    case 'taskPicker/setConstraints': {
      const base = state.taskPicker ?? DEFAULT_TASK_PICKER_CONSTRAINTS;
      return { ...state, taskPicker: { ...base, ...action.patch } };
    }
    case 'clear':
      return {};
    case 'restore':
      return action.state;
    case 'hardwareSizing/patch': {
      const merged: HardwareSizingState = { ...(state.hardwareSizing ?? DEFAULT_HARDWARE_SIZING), ...action.patch };
      // Explicitly clearing an optional field drops the key rather than keeping an undefined,
      // matching the Calculator reducer's handling of its own optional fields (see state.ts).
      if ('modelId' in action.patch && action.patch.modelId === undefined) delete merged.modelId;
      if ('maxTtftSeconds' in action.patch && action.patch.maxTtftSeconds === undefined) delete merged.maxTtftSeconds;
      if ('vendor' in action.patch && action.patch.vendor === undefined) delete merged.vendor;
      return { ...state, hardwareSizing: merged };
    }
  }
}

// ---------- URL persistence for the hardwareSizing slice (issue #25) ----------
// New keys only, all under the `ph` prefix so they can never collide with the Calculator's own
// keys (urlState.ts), the `tab` key, or #24's `pt`-prefixed TaskPicker keys. Every field is
// optional/defaulted on decode, and nothing is written until the user has actually touched step
// 2 (`hardwareSizing` is undefined until the first patch) — so a plain Calculator link, an old
// c2/c3 compare link, and a fresh Planner link that never opens step 2 all stay exactly as short
// as they are today.

const PH = {
  modelId: 'phm',
  useCalculatorModel: 'phcm',
  concurrentUsers: 'phu',
  contextTokens: 'phc',
  weightQuant: 'phwq',
  kvQuant: 'phkq',
  runtime: 'phrt',
  minPerUserTokS: 'phmt',
  maxTtftSeconds: 'phtt',
  vendor: 'phv',
  offloadEnabled: 'phof',
  sort: 'phs',
} as const;

/** Writes `state`'s fields onto `params` under the `ph` prefix; a no-op when `state` is undefined. */
export function encodeHardwareSizingParams(params: URLSearchParams, state: HardwareSizingState | undefined): void {
  if (!state) return;
  if (state.modelId !== undefined) params.set(PH.modelId, state.modelId);
  params.set(PH.useCalculatorModel, state.useCalculatorModel ? '1' : '0');
  params.set(PH.concurrentUsers, String(state.concurrentUsers));
  params.set(PH.contextTokens, String(state.contextTokens));
  params.set(PH.weightQuant, state.weightQuant);
  params.set(PH.kvQuant, state.kvQuant);
  params.set(PH.runtime, state.runtime);
  params.set(PH.minPerUserTokS, String(state.minPerUserTokS));
  if (state.maxTtftSeconds !== undefined) params.set(PH.maxTtftSeconds, String(state.maxTtftSeconds));
  if (state.vendor !== undefined) params.set(PH.vendor, state.vendor);
  params.set(PH.offloadEnabled, state.offloadEnabled ? '1' : '0');
  params.set(PH.sort, state.sort);
}

/**
 * Decodes the hardwareSizing slice from a query string (with or without the leading "?").
 * Undefined when the step's own marker key (`phu`) is absent, so every existing URL — including
 * a plain Calculator link and an old c2/c3 compare link — decodes unchanged.
 */
export function decodeHardwareSizingState(qs: string): HardwareSizingState | undefined {
  const q = new URLSearchParams(qs.startsWith('?') ? qs.slice(1) : qs);
  if (!q.has(PH.concurrentUsers)) return undefined;

  // Clamped the same way the Calculator's own inputs (and HardwareSizing.tsx's own NumberFields)
  // are: a crafted `phu`/`phc`/`phmt` (negative, zero, huge, or non-finite) must never reach
  // sizeHardware()'s calculate() calls unclamped (#27 review item 2).
  const numClamped = (k: string, fallback: number, lo: number, hi: number): number => {
    const raw = q.get(k);
    if (raw === null) return fallback;
    const n = Number(raw);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(hi, Math.max(lo, n));
  };
  const intClamped = (k: string, fallback: number, lo: number, hi: number): number => Math.round(numClamped(k, fallback, lo, hi));
  const oneOf = <T extends string>(k: string, allowed: readonly T[], fallback: T): T => {
    const raw = q.get(k);
    return raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;
  };

  const result: HardwareSizingState = {
    useCalculatorModel: q.get(PH.useCalculatorModel) === '1',
    concurrentUsers: intClamped(PH.concurrentUsers, DEFAULT_HARDWARE_SIZING.concurrentUsers, 1, MAX_USERS),
    contextTokens: intClamped(PH.contextTokens, DEFAULT_HARDWARE_SIZING.contextTokens, MIN_CONTEXT, MAX_CATALOG_CONTEXT),
    weightQuant: oneOf(PH.weightQuant, Object.keys(WEIGHT_QUANTS) as WeightQuantKey[], DEFAULT_HARDWARE_SIZING.weightQuant),
    kvQuant: oneOf(PH.kvQuant, Object.keys(KV_QUANTS) as KvQuantKey[], DEFAULT_HARDWARE_SIZING.kvQuant),
    runtime: oneOf(PH.runtime, RUNTIME_KEYS, DEFAULT_HARDWARE_SIZING.runtime),
    minPerUserTokS: numClamped(PH.minPerUserTokS, DEFAULT_HARDWARE_SIZING.minPerUserTokS, 0, MAX_MIN_PER_USER_TOK_S),
    offloadEnabled: q.get(PH.offloadEnabled) === '1',
    sort: oneOf(PH.sort, HARDWARE_SIZING_SORTS, DEFAULT_HARDWARE_SIZING.sort),
  };
  // A model id must resolve in the bundled catalog — an unrecognized or crafted id is dropped
  // rather than trusted, same as taskPickerUrl.ts validates its own gpu name.
  const modelId = q.get(PH.modelId);
  if (modelId !== null && findModelPreset(modelId)) result.modelId = modelId;
  if (q.has(PH.maxTtftSeconds)) {
    // Zero, negative or non-finite means "no cap" in the UI (the checkbox is unticked), so those
    // values are dropped rather than clamped to a fallback.
    const raw = q.get(PH.maxTtftSeconds);
    const n = raw === null ? NaN : Number(raw);
    if (Number.isFinite(n) && n > 0) result.maxTtftSeconds = Math.min(MAX_TTFT_SECONDS, n);
  }
  const vendorRaw = q.get(PH.vendor);
  if (vendorRaw !== null && GPU_VENDOR_GROUPS.some((v) => v.vendor === vendorRaw)) result.vendor = vendorRaw as GpuVendor;
  return result;
}
