import type { ModelSpec } from '../types';
import { MODEL_CATALOG } from './catalog';

/** Flat view of MODEL_CATALOG's specs, in catalog order. Existing preset ids/URLs and recents resolve against this. */
export const MODEL_PRESETS: ModelSpec[] = MODEL_CATALOG.map((e) => e.spec);

export function findModelPreset(idOrName: string): ModelSpec | undefined {
  const q = idOrName.trim().toLowerCase();
  return MODEL_PRESETS.find((m) => m.id.toLowerCase() === q || m.name.toLowerCase() === q);
}
