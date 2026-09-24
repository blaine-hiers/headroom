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
  /** Per GPU, dense (no sparsity) BF16 tensor TFLOPS; used for the prefill/TTFT estimate. */
  tflopsBf16: number;
  /** Percent of VRAM kept free (default 5). */
  reservePct: number;
  /** Runtime/CUDA-context overhead per GPU in GB (default 1). */
  overheadGB: number;
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
  totalBytes: number;
  headroomBytes: number; // usable - total (negative when it does not fit)
  fits: boolean;
  maxUsersAtContext: number;
  maxContextForUsers: number;
  /** rows for 2K / 8K / 32K / 128K plus the chosen context if different */
  contextTable: Array<{ contextTokens: number; kvBytesPerRequest: number; maxUsers: number }>;
  throughput: { perUserTokS: number; aggregateTokS: number; efficiency: number };
  /** Prefill / time-to-first-token estimate for one user at the chosen context (compute-bound). */
  prefill: { ttftSeconds: number; flops: number; mfu: number; headsSource: 'model' | 'hiddenSize-fallback' };
}
