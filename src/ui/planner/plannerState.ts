// Planner tab state. Deliberately separate from the Calculator's reducer (src/ui/state.ts) so
// switching tabs never loses either side's work: Calculator <-> Planner <-> Calculator round
// trips both untouched.
//
// Contract for #24 (TaskPicker) and #25 (HardwareSizing): each step gets its own optional
// sub-state slice here (`taskPicker` / `hardwareSizing`) plus its own action(s) below, instead of
// widening a shared shape. `handoffModelId` is the one field both steps already share — step 1
// ("Which model for this task?") sets it via `setHandoffModelId`, step 2 ("What hardware to
// serve N users?") reads it to preselect the model it sizes hardware for.

import { GPU_VENDOR_GROUPS, KV_QUANTS, RUNTIME_KEYS, WEIGHT_QUANTS } from '../../lib';
import type { GpuVendor, KvQuantKey, RuntimeKey, WeightQuantKey } from '../../lib';

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
};

export interface PlannerState {
  /** Set by TaskPicker (#24) once it recommends a model; read by HardwareSizing (#25). */
  handoffModelId?: string;
  /** #24 adds its own picker state here (selections, filters, etc.) as an optional sub-object. */
  taskPicker?: Record<string, never>;
  /** Step 2's inputs (#25); absent until the user first touches HardwareSizing. */
  hardwareSizing?: HardwareSizingState;
}

export const initialPlannerState: PlannerState = {};

export type PlannerAction =
  | { type: 'setHandoffModelId'; modelId: string | undefined }
  | { type: 'clear' }
  | { type: 'hardwareSizing/patch'; patch: Partial<HardwareSizingState> };

export function plannerReducer(state: PlannerState, action: PlannerAction): PlannerState {
  switch (action.type) {
    case 'setHandoffModelId': {
      if (action.modelId === undefined) {
        const { handoffModelId: _drop, ...rest } = state;
        return rest;
      }
      return { ...state, handoffModelId: action.modelId };
    }
    case 'clear':
      return {};
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
}

/**
 * Decodes the hardwareSizing slice from a query string (with or without the leading "?").
 * Undefined when the step's own marker key (`phu`) is absent, so every existing URL — including
 * a plain Calculator link and an old c2/c3 compare link — decodes unchanged.
 */
export function decodeHardwareSizingState(qs: string): HardwareSizingState | undefined {
  const q = new URLSearchParams(qs.startsWith('?') ? qs.slice(1) : qs);
  if (!q.has(PH.concurrentUsers)) return undefined;

  const num = (k: string, fallback: number): number => {
    const raw = q.get(k);
    if (raw === null) return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };
  const oneOf = <T extends string>(k: string, allowed: readonly T[], fallback: T): T => {
    const raw = q.get(k);
    return raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;
  };

  const result: HardwareSizingState = {
    useCalculatorModel: q.get(PH.useCalculatorModel) === '1',
    concurrentUsers: num(PH.concurrentUsers, DEFAULT_HARDWARE_SIZING.concurrentUsers),
    contextTokens: num(PH.contextTokens, DEFAULT_HARDWARE_SIZING.contextTokens),
    weightQuant: oneOf(PH.weightQuant, Object.keys(WEIGHT_QUANTS) as WeightQuantKey[], DEFAULT_HARDWARE_SIZING.weightQuant),
    kvQuant: oneOf(PH.kvQuant, Object.keys(KV_QUANTS) as KvQuantKey[], DEFAULT_HARDWARE_SIZING.kvQuant),
    runtime: oneOf(PH.runtime, RUNTIME_KEYS, DEFAULT_HARDWARE_SIZING.runtime),
    minPerUserTokS: num(PH.minPerUserTokS, DEFAULT_HARDWARE_SIZING.minPerUserTokS),
    offloadEnabled: q.get(PH.offloadEnabled) === '1',
  };
  const modelId = q.get(PH.modelId);
  if (modelId !== null) result.modelId = modelId;
  if (q.has(PH.maxTtftSeconds)) {
    const maxTtft = num(PH.maxTtftSeconds, NaN);
    if (Number.isFinite(maxTtft)) result.maxTtftSeconds = maxTtft;
  }
  const vendorRaw = q.get(PH.vendor);
  if (vendorRaw !== null && GPU_VENDOR_GROUPS.some((v) => v.vendor === vendorRaw)) result.vendor = vendorRaw as GpuVendor;
  return result;
}
