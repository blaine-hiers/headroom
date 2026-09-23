import { WEIGHT_QUANTS } from './quant';
import type { MoeSpec, WeightQuantKey } from './types';

/** Weight bytes = params × bitsPerWeight / 8. */
export function weightBytes(params: number, weightQuant: WeightQuantKey): number {
  return (params * WEIGHT_QUANTS[weightQuant].bitsPerWeight) / 8;
}

/**
 * Parameters touched per token. Dense: all of them. MoE estimate:
 * params × (expertsPerToken + sharedExperts) / (numExperts + sharedExperts).
 * Attention/embedding params are always active, so this understates slightly.
 */
export function activeParams(params: number, moe?: MoeSpec): number {
  if (!moe || moe.numExperts <= 1) return params;
  const active = moe.expertsPerToken + moe.sharedExperts;
  const total = moe.numExperts + moe.sharedExperts;
  return params * Math.min(1, active / total);
}
