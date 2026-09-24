// Prefill / time-to-first-token (TTFT) estimate. Prefill is compute-bound (unlike decode,
// which is bandwidth-bound — see throughput.ts), so it is estimated from FLOPs and the
// GPU's dense BF16 tensor rate rather than memory bandwidth.

/** Fraction of peak FLOPS a real kernel actually achieves. Real-world prefill runs are
 *  typically 30-50% of peak; 0.4 is a labelled, editable-in-spirit midpoint default. */
export const DEFAULT_MFU = 0.4;

export interface PrefillFlopsInput {
  /** Parameters touched per token (= CalcResult.activeParams; full params for dense models). */
  activeParams: number;
  promptTokens: number;
  numLayers: number;
  /** GQA/MHA head dim (ModelSpec.headDim); reused as-is for the attention term. */
  headDim: number;
  /** Query head count (ffn.numAttentionHeads). Optional: not every ModelSpec carries it. */
  numHeads?: number;
  /** Used as a stand-in for numHeads × headDim when numHeads is unknown. */
  hiddenSize: number;
}

export interface PrefillFlopsResult {
  flops: number;
  /** Whether the attention term used the real head count or fell back to hiddenSize. */
  headsSource: 'model' | 'hiddenSize-fallback';
}

/**
 * prefillFlops ≈ 2 × activeParams × promptTokens (the matmuls, same accounting as decode)
 * + 2 × numLayers × promptTokens² × numHeads × headDim (the O(n²) attention term: QKᵀ and
 * attention × V). `numHeads × headDim` is the query projection width; when the head count
 * isn't known (ffn.numAttentionHeads is optional), hiddenSize is a reasonable stand-in since
 * numHeads × headDim ≈ hiddenSize for the query projection in standard transformer layers.
 */
export function prefillFlops(input: PrefillFlopsInput): PrefillFlopsResult {
  const n = Math.max(0, input.promptTokens);
  const activeParams = Math.max(0, input.activeParams);
  const numLayers = Math.max(0, input.numLayers);
  const matmul = 2 * activeParams * n;
  const hasHeads = input.numHeads !== undefined && Number.isFinite(input.numHeads) && input.numHeads > 0;
  const headsSource: PrefillFlopsResult['headsSource'] = hasHeads ? 'model' : 'hiddenSize-fallback';
  const queryWidth = hasHeads ? (input.numHeads as number) * Math.max(0, input.headDim) : Math.max(0, input.hiddenSize);
  const attention = 2 * numLayers * n * n * queryWidth;
  return { flops: matmul + attention, headsSource };
}

export interface TtftInput {
  flops: number;
  /** Per GPU, dense (no sparsity) BF16 TFLOPS. Keep the bf16 rate unless an fp8 rate is added. */
  tflopsBf16: number;
  gpuCount: number;
  /** Fraction of peak FLOPS achieved; defaults to DEFAULT_MFU. */
  mfu?: number;
}

export interface TtftResult {
  ttftSeconds: number;
  mfu: number;
}

/** TTFT = prefillFlops / (tflops × 1e12 × gpuCount × MFU). 0 (not NaN/Infinity) when degenerate. */
export function estimateTtft(input: TtftInput): TtftResult {
  const mfu = input.mfu ?? DEFAULT_MFU;
  const tflops = Math.max(0, input.tflopsBf16);
  const gpuCount = Math.max(0, input.gpuCount);
  const denom = tflops * 1e12 * gpuCount * mfu;
  const ttftSeconds = denom > 0 && input.flops > 0 ? input.flops / denom : 0;
  return { ttftSeconds, mfu };
}
