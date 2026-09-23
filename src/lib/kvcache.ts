import { KV_QUANTS } from './quant';
import type { KvQuantKey, ModelSpec } from './types';

/**
 * KV-cache bytes one token occupies in one layer.
 * GQA/MHA: 2 (K and V) × numKvHeads × headDim × bytes.
 * MLA: one compressed latent (kvLoraRank + qkRopeHeadDim) × bytes, no ×2.
 */
export function kvBytesPerTokenPerLayer(spec: ModelSpec, kv: KvQuantKey): number {
  const bytes = KV_QUANTS[kv].bytesPerElement;
  if (spec.attention === 'mla') {
    return ((spec.kvLoraRank ?? 0) + (spec.qkRopeHeadDim ?? 0)) * bytes;
  }
  return 2 * spec.numKvHeads * spec.headDim * bytes;
}

/** Full-attention rate: per-layer bytes × numLayers (ignores sliding-window caps). */
export function kvBytesPerToken(spec: ModelSpec, kv: KvQuantKey): number {
  return kvBytesPerTokenPerLayer(spec, kv) * spec.numLayers;
}

/** Number of sliding-window layers actually applied (0 when there is no usable window). */
export function effectiveSlidingLayers(spec: ModelSpec): number {
  if (!spec.slidingWindow || spec.slidingWindow <= 0) return 0;
  const n = spec.slidingLayers ?? 0;
  return Math.max(0, Math.min(n, spec.numLayers));
}

/** KV bytes for one request holding `contextTokens`; sliding layers cap at the window. */
export function kvBytesForContext(spec: ModelSpec, contextTokens: number, kv: KvQuantKey): number {
  const ctx = Math.max(0, contextTokens);
  const perLayer = kvBytesPerTokenPerLayer(spec, kv);
  const sliding = effectiveSlidingLayers(spec);
  const full = spec.numLayers - sliding;
  const slidingTokens = sliding > 0 ? Math.min(ctx, spec.slidingWindow ?? 0) : 0;
  return perLayer * (full * ctx + sliding * slidingTokens);
}
