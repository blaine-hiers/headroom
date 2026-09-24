import { WEIGHT_QUANTS } from './quant';
import type { ModelSpec, WeightQuantKey, WeightSource } from './types';
import { weightBytes } from './weights';

export interface ResolvedWeights {
  bytes: number;
  /** Weight bytes for `activeParams`: the same share of the file bytes, or the estimate. */
  activeBytes: number;
  source: WeightSource;
}

/** Effective bits per weight of a byte count: bytes × 8 / params (0 when params is unknown). */
export function effectiveBitsPerWeight(bytes: number, params: number): number {
  return params > 0 ? (bytes * 8) / params : 0;
}

/**
 * Exact file bytes when the spec has them and the chosen quant is the one the files are in;
 * otherwise the estimate, params × bitsPerWeight / 8. Active bytes scale the file bytes by
 * activeParams / params (all of them when params is unknown).
 */
export function resolveWeights(model: ModelSpec, quant: WeightQuantKey, activeParams: number): ResolvedWeights {
  const files = model.fileWeights;
  if (files && files.bytes > 0 && files.quant === quant) {
    const share = model.params > 0 ? Math.min(1, activeParams / model.params) : 1;
    return { bytes: files.bytes, activeBytes: files.bytes * share, source: 'files' };
  }
  return { bytes: weightBytes(model.params, quant), activeBytes: weightBytes(activeParams, quant), source: 'estimate' };
}

/** The table quant whose bits per weight is nearest `bits` (the first one on a tie). */
export function closestWeightQuant(bits: number): WeightQuantKey | undefined {
  if (!(bits > 0)) return undefined;
  let best: WeightQuantKey | undefined;
  let bestDiff = Infinity;
  for (const [key, info] of Object.entries(WEIGHT_QUANTS) as Array<[WeightQuantKey, { bitsPerWeight: number }]>) {
    const diff = Math.abs(info.bitsPerWeight - bits);
    if (diff < bestDiff) {
      best = key;
      bestDiff = diff;
    }
  }
  return best;
}
