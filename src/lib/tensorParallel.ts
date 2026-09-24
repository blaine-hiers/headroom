import type { ModelSpec, TensorParallelCheck } from './types';

/** Upper bound for the GPU-count search when `numAttentionHeads` isn't known to cap it. */
const SEARCH_CAP = 64;

/** Whether `gpuCount` GPUs can hold `kvHeads` KV heads evenly: split (kvHeads % g === 0) or replicate (g % kvHeads === 0). */
function kvSplitValidAt(kvHeads: number, g: number): boolean {
  if (kvHeads <= 0 || g <= 0) return true;
  return kvHeads >= g ? kvHeads % g === 0 : g % kvHeads === 0;
}

/**
 * GPU counts (other than `near`) that divide `heads` evenly (when known) AND lay out
 * `kvHeads` evenly (when given), closest to `near` first, capped at `limit`, ascending.
 */
function nearestValidGpuCounts(near: number, heads: number | undefined, kvHeads: number | undefined, limit = 4): number[] {
  const upper = heads ?? SEARCH_CAP;
  const candidates: number[] = [];
  for (let g = 1; g <= upper; g++) {
    if (heads !== undefined && heads % g !== 0) continue;
    if (kvHeads !== undefined && !kvSplitValidAt(kvHeads, g)) continue;
    candidates.push(g);
  }
  return candidates
    .filter((g) => g !== near)
    .sort((a, b) => Math.abs(a - near) - Math.abs(b - near) || a - b)
    .slice(0, limit)
    .sort((a, b) => a - b);
}

export function checkTensorParallelSplit(model: ModelSpec, gpuCount: number): TensorParallelCheck {
  const n = Math.max(1, Math.floor(gpuCount));
  const heads = model.ffn?.numAttentionHeads;
  const checkable = heads !== undefined && heads > 0;
  const headsDivisible = n <= 1 || !checkable || heads % n === 0;

  // MLA stores one compressed KV latent per token — there are no per-head KV projections to
  // shard or replicate, so KV-head replication and its validity check never apply to it.
  const isMla = model.attention === 'mla';
  const kvHeads = model.numKvHeads;
  const kvCheckable = !isMla && n > 1 && kvHeads > 0;
  const kvHeadsReplicated = kvCheckable && kvHeads < n;
  // vLLM lays KV heads out evenly only when kvHeads divides gpuCount (split) or gpuCount
  // divides kvHeads (replicate); e.g. 3 KV heads across 8 GPUs satisfies neither.
  const kvHeadsSplitValid = !kvCheckable || kvSplitValidAt(kvHeads, n);
  const effectiveKvHeads = kvHeadsReplicated ? n : kvHeads;
  // Total KV heads resident across all GPUs when each GPU holds at least one head, i.e. the
  // worst-case per-GPU memory multiplier — reported regardless of whether the split is even.
  const kvReplicationFactor = kvCheckable && kvHeadsReplicated ? effectiveKvHeads / kvHeads : 1;

  const suggestedGpuCounts =
    (checkable && !headsDivisible) || (kvCheckable && !kvHeadsSplitValid)
      ? nearestValidGpuCounts(n, checkable ? heads : undefined, kvCheckable ? kvHeads : undefined)
      : [];

  return {
    gpuCount: n,
    numAttentionHeads: heads,
    checkable,
    headsDivisible,
    suggestedGpuCounts,
    kvHeadsReplicated,
    kvHeadsSplitValid,
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
