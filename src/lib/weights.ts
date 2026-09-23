import { WEIGHT_QUANTS } from './quant';
import type { ActiveParamsMethod, ModelSpec, MoeSpec, WeightQuantKey } from './types';

/** Weight bytes = params × bitsPerWeight / 8. */
export function weightBytes(params: number, weightQuant: WeightQuantKey): number {
  return (params * WEIGHT_QUANTS[weightQuant].bitsPerWeight) / 8;
}

/**
 * Ratio fallback for MoE active params: params × (expertsPerToken + sharedExperts) /
 * (numExperts + sharedExperts). Attention and embeddings are always active, so this
 * can understate the real figure by a third. Dense: all of them.
 */
export function activeParams(params: number, moe?: MoeSpec): number {
  if (!moe || moe.numExperts <= 1) return params;
  const active = moe.expertsPerToken + moe.sharedExperts;
  const total = moe.numExperts + moe.sharedExperts;
  return params * Math.min(1, active / total);
}

export interface ActiveParamsDetail {
  active: number;
  method: ActiveParamsMethod;
}

const pos = (v: number | undefined): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0;

/** Attention weights for one layer (q, k, v, o projections; biases and norms ignored). */
export function attentionParamsPerLayer(spec: ModelSpec): number | undefined {
  const ffn = spec.ffn;
  if (!ffn) return undefined;
  const h = spec.hiddenSize;
  const heads = ffn.numAttentionHeads;
  if (!pos(h) || !pos(heads)) return undefined;
  if (spec.attention === 'mla' && pos(spec.kvLoraRank) && pos(ffn.vHeadDim)) {
    // DeepSeek-style MLA: low-rank q (optional) and a shared kv latent.
    const rope = spec.qkRopeHeadDim ?? 0;
    const qkHead = spec.headDim; // qk_nope_head_dim + qk_rope_head_dim
    const c = spec.kvLoraRank;
    const nope = Math.max(0, qkHead - rope);
    const v = ffn.vHeadDim;
    const q = pos(ffn.qLoraRank) ? h * ffn.qLoraRank + ffn.qLoraRank * heads * qkHead : h * heads * qkHead;
    return q + h * (c + rope) + c * heads * (nope + v) + heads * v * h;
  }
  const hd = spec.headDim;
  if (!pos(hd)) return undefined;
  const qDim = heads * hd;
  const kvDim = spec.numKvHeads * hd;
  return h * qDim + 2 * h * kvDim + qDim * h;
}

/**
 * Structural active-params estimate from the layer shapes. Only the output head counts
 * for the embeddings: the input embedding is a one-row lookup per token, so it is not
 * streamed from memory each step (tied or not, the head is read once).
 */
function structuralActive(spec: ModelSpec, moe: MoeSpec): number | undefined {
  const ffn = spec.ffn;
  const attn = attentionParamsPerLayer(spec);
  const h = spec.hiddenSize;
  const layers = spec.numLayers;
  if (!ffn || attn === undefined || !pos(layers)) return undefined;
  const expertSize = ffn.moeIntermediateSize ?? ffn.intermediateSize;
  if (!pos(expertSize)) return undefined;
  const dense = Math.min(layers, Math.max(0, Math.floor(ffn.firstKDense ?? 0)));
  const denseFfn = 3 * h * (ffn.intermediateSize > 0 ? ffn.intermediateSize : 0);
  const moeFfn = 3 * h * expertSize * (moe.expertsPerToken + moe.sharedExperts) + h * moe.numExperts;
  const head = Math.max(0, spec.vocabSize) * h;
  return layers * attn + dense * denseFfn + (layers - dense) * moeFfn + head;
}

/**
 * Parameters touched per token, and how the figure was reached: dense models use all
 * params; MoE uses the structural estimate when the FFN shapes are known, else the ratio.
 * The structural figure never exceeds the total params when those are known.
 */
export function activeParamsDetailed(spec: ModelSpec): ActiveParamsDetail {
  const moe = spec.moe;
  if (!moe || moe.numExperts <= 1) return { active: spec.params, method: 'dense' };
  const structural = structuralActive(spec, moe);
  if (structural !== undefined && structural > 0) {
    return { active: spec.params > 0 ? Math.min(spec.params, structural) : structural, method: 'structural' };
  }
  return { active: activeParams(spec.params, moe), method: 'ratio' };
}
