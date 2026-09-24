import type { WeightQuantKey } from './types';
import { weightBytes } from './weights';
import { catalogByProvider, type CatalogEntry } from './presets/catalog';

export interface ProviderModelRow {
  entry: CatalogEntry;
  /** Weight bytes at `weightQuant`, from the same maths as the rest of the calculator (weightBytes). */
  weightBytes: number;
  /** True when this row's model id is the currently loaded model. */
  isLoaded: boolean;
}

/**
 * One provider's catalog entries (see `catalogByProvider`), each paired with its weight
 * size at `weightQuant` and whether it's the model currently loaded, in catalog order.
 * Powers the provider picker's model list.
 */
export function providerModelRows(providerId: string, weightQuant: WeightQuantKey, loadedModelId: string): ProviderModelRow[] {
  return catalogByProvider(providerId).map((entry) => ({
    entry,
    weightBytes: weightBytes(entry.spec.params, weightQuant),
    isLoaded: entry.spec.id === loadedModelId,
  }));
}
