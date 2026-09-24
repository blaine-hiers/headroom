// Numeric bounds the Calculator's own input clamps (src/ui/state.ts) already enforce, plus
// anything in src/lib that needs the same bounds without importing from src/ui — e.g.
// taskPickerUrl.ts clamping a decoded URL value before it ever reaches `calculate()`.
// src/ui/state.ts re-exports these so every existing `import { MAX_GPUS } from './state'` (etc.)
// keeps working unchanged.

import { MODEL_CATALOG } from './presets/catalog';

export const MIN_CONTEXT = 256;
export const MAX_USERS = 512;
export const MAX_GPUS = 16;

/** The longest context any bundled catalog model supports — the upper clamp for a context input not yet tied to one specific model (e.g. the TaskPicker's "context needed"). */
export const MAX_CATALOG_CONTEXT = Math.max(...MODEL_CATALOG.map((e) => e.spec.maxPositionEmbeddings));
