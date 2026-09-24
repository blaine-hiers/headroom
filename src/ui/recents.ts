import type { ModelSpec } from '../lib';

const RECENTS_KEY = 'headroom.recents';
const MAX_RECENTS = 8;

export function getRecents(): ModelSpec[] {
  try {
    const json = window.localStorage.getItem(RECENTS_KEY);
    if (!json) return [];
    const recents = JSON.parse(json) as unknown;
    if (!Array.isArray(recents)) return [];
    // Validate that each item is a ModelSpec-like object
    return recents.filter(isModelSpec);
  } catch {
    return [];
  }
}

function isModelSpec(obj: unknown): obj is ModelSpec {
  return (
    obj !== null &&
    typeof obj === 'object' &&
    'id' in obj &&
    'name' in obj &&
    'params' in obj &&
    'activeParams' in obj &&
    'numLayers' in obj &&
    'attention' in obj &&
    'numKvHeads' in obj &&
    'headDim' in obj &&
    'maxPositionEmbeddings' in obj &&
    'hiddenSize' in obj &&
    'vocabSize' in obj &&
    'nativeDtype' in obj &&
    'source' in obj &&
    'warnings' in obj &&
    Array.isArray((obj as ModelSpec).warnings)
  );
}

export function addRecent(spec: ModelSpec): void {
  try {
    const recents = getRecents();
    // Remove if already present
    const filtered = recents.filter((m) => m.id !== spec.id);
    // Add to front
    const updated = [spec, ...filtered].slice(0, MAX_RECENTS);
    window.localStorage.setItem(RECENTS_KEY, JSON.stringify(updated));
  } catch {
    // ignore: persistence is a convenience
  }
}

export function removeRecent(id: string): void {
  try {
    const recents = getRecents();
    const updated = recents.filter((m) => m.id !== id);
    if (updated.length === 0) {
      window.localStorage.removeItem(RECENTS_KEY);
    } else {
      window.localStorage.setItem(RECENTS_KEY, JSON.stringify(updated));
    }
  } catch {
    // ignore: persistence is a convenience
  }
}

export function clearRecents(): void {
  try {
    window.localStorage.removeItem(RECENTS_KEY);
  } catch {
    // ignore: persistence is a convenience
  }
}
