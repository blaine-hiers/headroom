import type { ModelSpec, TensorParallelCheck } from './types';

/** Divisors of `heads`, closest to `near` first (ties broken by smaller value), capped at `limit`. */
function nearestDivisors(heads: number, near: number, limit = 4): number[] {
  const divisors: number[] = [];
  for (let d = 1; d <= heads; d++) {
    if (heads % d === 0) divisors.push(d);
  }
  return divisors
    .filter((d) => d !== near)
    .sort((a, b) => Math.abs(a - near) - Math.abs(b - near) || a - b)
    .slice(0, limit)
    .sort((a, b) => a - b);
}

export function checkTensorParallelSplit(model: ModelSpec, gpuCount: number): TensorParallelCheck {
  const n = Math.max(1, Math.floor(gpuCount));
  const heads = model.ffn?.numAttentionHeads;
  const checkable = heads !== undefined && heads > 0;
  const headsDivisible = n <= 1 || !checkable || heads % n === 0;
  const suggestedGpuCounts = checkable && !headsDivisible ? nearestDivisors(heads, n) : [];

  const kvHeads = model.numKvHeads;
  const kvHeadsReplicated = n > 1 && kvHeads > 0 && kvHeads < n;
  const effectiveKvHeads = kvHeadsReplicated ? n : kvHeads;
  const kvReplicationFactor = kvHeads > 0 ? effectiveKvHeads / kvHeads : 1;

  return {
    gpuCount: n,
    numAttentionHeads: heads,
    checkable,
    headsDivisible,
    suggestedGpuCounts,
    kvHeadsReplicated,
    effectiveKvHeads,
    kvReplicationFactor,
  };
}

/** Efficiency penalty per doubling of GPU count, applied by tensorParallelEfficiency. */
export const TP_PENALTY_PER_DOUBLING = 0.9;

/**
 * Small labelled scaling penalty for tensor-parallel communication overhead:
 * ×TP_PENALTY_PER_DOUBLING for each doubling of GPU count. 1 at gpuCount ≤ 1,
 * so single-GPU numbers are unaffected.
 */
export function tensorParallelEfficiency(gpuCount: number): number {
  const n = Math.max(1, Math.floor(gpuCount));
  if (n <= 1) return 1;
  return TP_PENALTY_PER_DOUBLING ** Math.log2(n);
}
