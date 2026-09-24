// Numeric bounds the Calculator's own input clamps (src/ui/state.ts) already enforce, plus
// anything in src/lib that needs the same bounds without importing from src/ui — e.g.
// taskPickerUrl.ts clamping a decoded URL value before it ever reaches `calculate()`.
// src/ui/state.ts re-exports these so every existing `import { MAX_GPUS } from './state'` (etc.)
// keeps working unchanged.

import { MODEL_CATALOG } from './presets/catalog';
import type { ModelSpec, Workload } from './types';

export const MIN_CONTEXT = 256;
export const MAX_USERS = 512;
export const MAX_GPUS = 16;

/** The longest context any bundled catalog model supports — the upper clamp for a context input not yet tied to one specific model (e.g. the TaskPicker's "context needed"). */
export const MAX_CATALOG_CONTEXT = Math.max(...MODEL_CATALOG.map((e) => e.spec.maxPositionEmbeddings));

function clampNum(n: number, lo: number, hi: number): number {
  if (!Number.isFinite(n)) return lo;
  return Math.min(hi, Math.max(lo, n));
}

/** Upper bound of the context control for a model (never below the minimum). */
export function maxContextFor(model: ModelSpec): number {
  return Math.max(MIN_CONTEXT, Math.floor(model.maxPositionEmbeddings) || MIN_CONTEXT);
}

/**
 * Clamps a workload to what "Use" will actually load for this model: context to
 * [MIN_CONTEXT, this model's own maxPositionEmbeddings], users to [1, MAX_USERS]. Shared by
 * src/ui/state.ts's clampWorkload (the Calculator's own clamp on "Use"), taskPicker.ts's
 * calcStateFor (#24), and hardwareSizing.ts's sizeHardware (#25) so a row's numbers always match
 * what "Use" loads instead of drifting when the typed workload exceeds the model's own max
 * context (see #27 review item 1).
 */
export function clampWorkloadFor(model: ModelSpec, workload: Workload): Workload {
  return {
    contextTokens: Math.round(clampNum(workload.contextTokens, MIN_CONTEXT, maxContextFor(model))),
    concurrentUsers: Math.round(clampNum(workload.concurrentUsers, 1, MAX_USERS)),
  };
}
