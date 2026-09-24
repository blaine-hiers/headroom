// Contract shared by lib/ and ui/. Keep framework-free.

export type Attention = 'mha_gqa' | 'mla';

/** How active params were reached: all params, layer shapes, or the MoE params ratio. */
export type ActiveParamsMethod = 'structural' | 'ratio' | 'dense';

export type NativeDtype = 'bf16' | 'fp16' | 'fp32' | 'fp8';

export interface MoeSpec {
  numExperts: number;
  expertsPerToken: number;
  sharedExperts: number;
}

/**
 * Layer shapes for the structural active-params estimate (optional: without them an MoE
 * model falls back to the params ratio).
 */
export interface FfnSpec {
  /** Dense FFN width (intermediate_size); for gpt-oss/Mixtral this is also the expert width. */
  intermediateSize: number;
  /** Expert FFN width (moe_intermediate_size), when it differs from intermediateSize. */
  moeIntermediateSize?: number;
  /** Leading dense layers before the MoE layers start (first_k_dense_replace). */
  firstKDense?: number;
  numAttentionHeads: number;
  tieEmbeddings: boolean;
  /** MLA only: q_lora_rank (absent = full-rank q projection). */
  qLoraRank?: number;
  /** MLA only: v_head_dim. */
  vHeadDim?: number;
}

export interface ModelSpec {
  id: string;
  name: string;
  /** Total parameters (safetensors total). */
  params: number;
  /** Parameters touched per token: = params for dense, estimate for MoE. */
  activeParams: number;
  numLayers: number;
  attention: Attention;
  /** GQA/MHA fields */
  numKvHeads: number;
  headDim: number;
  /** MLA fields */
  kvLoraRank?: number;
  qkRopeHeadDim?: number;
  /** Sliding-window attention */
  slidingWindow?: number;
  /** How many of numLayers use the sliding window (0 = none). */
  slidingLayers?: number;
  maxPositionEmbeddings: number;
  hiddenSize: number;
  vocabSize: number;
  nativeDtype: NativeDtype;
  moe?: MoeSpec;
  ffn?: FfnSpec;
  source: 'hf' | 'preset' | 'manual';
  warnings: string[];
}

/**
 * Multi-GPU inference shards the model across GPUs with tensor parallelism.
 * vLLM (and most runtimes) refuse to start when numAttentionHeads isn't evenly
 * divisible by the GPU count, and when there are fewer KV heads than GPUs the
 * KV heads get replicated onto more than one GPU, raising the KV memory total.
 * None of the KV-head fields apply to MLA models: MLA caches one compressed
 * latent per token with no per-head KV projections to shard or replicate.
 */
export interface TensorParallelCheck {
  /** gpuCount this check was run for. */
  gpuCount: number;
  /** Attention head count used for the split check (ffn.numAttentionHeads), when known. */
  numAttentionHeads?: number;
  /** True when the head count is unknown (no ffn spec) — the split can't be checked. */
  checkable: boolean;
  /** True when gpuCount ≤ 1, headless, or numAttentionHeads % gpuCount === 0. */
  headsDivisible: boolean;
  /** Nearest GPU counts that satisfy both headsDivisible and kvHeadsSplitValid, ascending. */
  suggestedGpuCounts: number[];
  /** True when numKvHeads < gpuCount, so at least one KV head is replicated onto more than one GPU. Always false for MLA. */
  kvHeadsReplicated: boolean;
  /**
   * True when numKvHeads can be laid out evenly across gpuCount GPUs: numKvHeads % gpuCount === 0
   * (split) or gpuCount % numKvHeads === 0 (replicate) — e.g. 3 KV heads over 8 GPUs satisfies
   * neither and is flagged. Always true for MLA, gpuCount ≤ 1, or a missing/zero KV-head count.
   */
  kvHeadsSplitValid: boolean;
  /** KV heads actually resident across all GPUs once replication is accounted for (== numKvHeads unless replicated). */
  effectiveKvHeads: number;
  /**
   * effectiveKvHeads / numKvHeads: the per-GPU KV memory multiplier when each GPU holds at
   * least one KV head (gpuCount / numKvHeads), reported even when kvHeadsSplitValid is false.
   * 1 when there is no replication or the model is MLA. Multiplies KV bytes.
   */
  kvReplicationFactor: number;
}

export type DraftMode = 'none' | 'preset' | 'custom';

/**
 * Speculative decoding: a small draft model proposes k tokens per step, the target model
 * verifies them in one pass, and accepted tokens are kept. 'none' is n-gram / prompt-lookup
 * drafting (no model, no memory, no per-step compute cost).
 */
export interface SpeculativeConfig {
  enabled: boolean;
  draftMode: DraftMode;
  /** draftMode 'preset': the draft's own ModelSpec, so its KV cache can be estimated exactly. */
  draftModel?: ModelSpec;
  /** draftMode 'custom': parameter count only. KV cache is not estimated (architecture unknown). */
  draftParams?: number;
  draftWeightQuant: WeightQuantKey;
  /** Draft tokens proposed per verify step. */
  k: number;
  /** Expected per-token acceptance probability (workload-dependent). */
  alpha: number;
}

export interface SpeculativeMemory {
  /** Draft weights resident in VRAM (all params, at draftWeightQuant). 0 when disabled or draftMode 'none'. */
  weightBytes: number;
  /** Draft weights read per decode step (handles MoE active params for preset drafts). */
  activeWeightBytes: number;
  /** Draft KV cache for one request at the target's context; 0 unless draftMode 'preset'. */
  kvBytesPerRequest: number;
  kvBytesAllUsers: number;
  /** Draft KV cache per token, full-attention rate; 0 unless draftMode 'preset'. Feeds maxContextForUsers. */
  kvBytesPerToken: number;
  totalBytes: number;
}

export interface SpeculativeThroughput {
  /** (1 − α^(k+1)) / (1 − α); the removable α = 1 singularity resolves to k + 1. */
  expectedTokensPerStep: number;
  targetStepSeconds: number;
  draftStepSeconds: number;
  verifyStepSeconds: number;
  perUserTokS: number;
  aggregateTokS: number;
  /** perUserTokS vs the no-speculation baseline; 1 when disabled. */
  multiplier: number;
}

export interface SpeculativeResult {
  enabled: boolean;
  memory: SpeculativeMemory;
  throughput: SpeculativeThroughput;
}

export type WeightQuantKey =
  | 'fp32'
  | 'bf16'
  | 'fp16'
  | 'fp8'
  | 'q8_0'
  | 'q6_k'
  | 'q5_k_m'
  | 'q4_k_m'
  | 'q4_0'
  | 'awq_gptq_4bit'
  | 'iq4_xs'
  | 'nf4'
  | 'q3_k_m'
  | 'q2_k';

export type KvQuantKey = 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4';

export interface Quant {
  weight: WeightQuantKey;
  kv: KvQuantKey;
}

export interface HardwareSpec {
  gpuName: string;
  gpuCount: number;
  /** Per GPU, vendor-style decimal GB. */
  vramGB: number;
  /** Per GPU, GB/s. */
  bandwidthGBs: number;
  /** Percent of VRAM kept free (default 5). */
  reservePct: number;
  /** Runtime/CUDA-context overhead per GPU in GB (default 1). */
  overheadGB: number;
  /** Apple GPU wired memory limit per GPU in GB (only applies to Apple GPUs). Optional; when omitted, the default OS limit is applied. */
  appleWiredLimitGB?: number;
}

export interface Workload {
  contextTokens: number;
  concurrentUsers: number;
}

export interface CalcState {
  model: ModelSpec;
  quant: Quant;
  hardware: HardwareSpec;
  workload: Workload;
  /** Optional so existing states/URLs decode unchanged; treated as disabled when absent. */
  speculative?: SpeculativeConfig;
}

/** Everything the results column needs; all sizes in bytes. */
export interface CalcResult {
  kvBytesPerToken: number; // full-attention rate (no sliding cap)
  kvBytesPerRequest: number; // at workload.contextTokens, sliding-window aware
  kvBytesAllUsers: number;
  weightBytes: number;
  /** Parameters read per decode step, and how the figure was estimated. */
  activeParams: number;
  activeParamsMethod: ActiveParamsMethod;
  overheadBytes: number;
  usableBytes: number;
  /**
   * Everything that doesn't scale with users: weights + overhead + the draft model's weights
   * (0 when speculation is off). Equals `fixed` in fit.ts; what Chart.tsx's line starts from.
   */
  fixedBytes: number;
  /**
   * Everything that scales per user at the chosen context: target KV per request + the
   * draft's KV per request (0 when speculation is off). Chart.tsx's line slope.
   */
  bytesPerUser: number;
  totalBytes: number;
  headroomBytes: number; // usable - total (negative when it does not fit)
  fits: boolean;
  maxUsersAtContext: number;
  maxContextForUsers: number;
  /** rows for 2K / 8K / 32K / 128K plus the chosen context if different */
  contextTable: Array<{ contextTokens: number; kvBytesPerRequest: number; maxUsers: number }>;
  throughput: { perUserTokS: number; aggregateTokS: number; efficiency: number };
  /** Tensor-parallel split check for hardware.gpuCount (see tensorParallel.ts). */
  tensorParallel: TensorParallelCheck;
  /** Speculative-decoding memory and speedup estimate (see speculative.ts). enabled: false when off/absent. */
  speculative: SpeculativeResult;
}
