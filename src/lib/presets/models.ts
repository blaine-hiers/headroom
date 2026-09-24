import type { ModelSpec } from '../types';
import { MODEL_CATALOG } from './catalog';

/** Flat view of MODEL_CATALOG's specs, in catalog order. Existing preset ids/URLs and recents resolve against this. */
export const MODEL_PRESETS: ModelSpec[] = MODEL_CATALOG.map((e) => e.spec);

export function findModelPreset(idOrName: string): ModelSpec | undefined {
  const q = idOrName.trim().toLowerCase();
  return MODEL_PRESETS.find((m) => m.id.toLowerCase() === q || m.name.toLowerCase() === q);
}

/**
 * The catalog's fallback model for a picker with no better source — an explicit id rather than
 * `MODEL_PRESETS[0]` (array position), since the catalog (#22) reordered the array and silently
 * changed `MODEL_PRESETS[0]` from Llama 3.1 8B to Llama 3.2 1B, which in turn changed the
 * pre-existing speculative-draft default (SpeculativeFields.tsx) and would have changed step 2's
 * hardware-sizing default (HardwareSizing.tsx, #25) too (#27 review). Falls back to
 * `MODEL_PRESETS[0]` only if this id is ever removed from the bundled catalog.
 */
export const DEFAULT_MODEL_PRESET: ModelSpec = findModelPreset('meta-llama/Llama-3.1-8B-Instruct') ?? MODEL_PRESETS[0];
