import { deriveWarnings, VISION_WARNING } from '../hf';
import type { ModelSpec } from '../types';
import { activeParamsDetailed } from '../weights';

// Values checked 2026-09-24 against each repo's config.json and the Hub API's
// safetensors.total (gated Meta/Google repos: API param count + published config values,
// carried over from the original hand-verified presets). `npm run check-catalog` re-fetches
// the ungated entries and diffs them against what is bundled here.

type PresetInput = Omit<ModelSpec, 'activeParams' | 'source' | 'warnings'> & { warnings?: string[] };

/** Build a full ModelSpec the way the Hub fetch path (parseConfig) would, so the calculator sees the same shape offline. */
function preset(p: PresetInput): ModelSpec {
  const spec: ModelSpec = { slidingLayers: 0, ...p, activeParams: 0, source: 'preset', warnings: [] };
  spec.activeParams = activeParamsDetailed(spec).active;
  spec.warnings = [...deriveWarnings(spec), ...(p.warnings ?? [])];
  return spec;
}

const MXFP4_NOTE = 'weights ship as MXFP4 (~4.25 bits/weight for the experts); pick a 4-bit weight quant to match the download size';

/** Fixed task/capability vocabulary. Coarse tags taken from each model card's own stated capabilities — not a benchmark. */
export const TASK_TAGS = [
  'chat',
  'coding',
  'reasoning',
  'long-context',
  'vision',
  'tool-use',
  'multilingual',
  'small-edge',
] as const;

export type TaskTag = (typeof TASK_TAGS)[number];

export interface Provider {
  id: string;
  /** Display name. */
  name: string;
  /** Hugging Face Hub org this provider's repos live under. */
  hubOrg: string;
}

/** Providers in display order. */
export const PROVIDERS: Provider[] = [
  { id: 'meta', name: 'Meta', hubOrg: 'meta-llama' },
  { id: 'qwen', name: 'Qwen', hubOrg: 'Qwen' },
  { id: 'google', name: 'Google', hubOrg: 'google' },
  { id: 'mistral', name: 'Mistral', hubOrg: 'mistralai' },
  { id: 'deepseek', name: 'DeepSeek', hubOrg: 'deepseek-ai' },
  { id: 'openai', name: 'OpenAI', hubOrg: 'openai' },
  { id: 'microsoft', name: 'Microsoft', hubOrg: 'microsoft' },
  { id: 'moonshot', name: 'Moonshot', hubOrg: 'moonshotai' },
  { id: 'zhipu', name: 'Zhipu', hubOrg: 'zai-org' },
  { id: 'nvidia', name: 'NVIDIA', hubOrg: 'nvidia' },
];

/**
 * Catalog-only metadata, kept separate from ModelSpec so preset ids/URLs never change shape.
 * `provider`/`family` are free text matching a PROVIDERS id and a human family name
 * ("Llama 3.x", "Qwen3", ...); `tags` are drawn from TASK_TAGS.
 */
export interface CatalogMeta {
  provider: string;
  family: string;
  tags: TaskTag[];
  /** Short SPDX-ish license id (e.g. "apache-2.0", "mit"), or the vendor's own community-license name when there is no SPDX id. */
  license: string;
  /** YYYY-MM, approximate. */
  releaseDate: string;
  note?: string;
}

export interface CatalogEntry {
  spec: ModelSpec;
  meta: CatalogMeta;
}

export const MODEL_CATALOG: CatalogEntry[] = [
  // ---------------------------------------------------------------------------------------
  // Meta — gated on the Hub (401 without a token). Values are the original hand-verified
  // presets (API param count + published config.json values), not re-fetched by the checker.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'meta', family: 'Llama 3.x', tags: ['chat', 'tool-use', 'multilingual', 'small-edge'], license: 'llama3.2', releaseDate: '2024-09' },
    spec: preset({
      id: 'meta-llama/Llama-3.2-1B-Instruct',
      name: 'Llama 3.2 1B',
      params: 1_235_814_400,
      numLayers: 16,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 64,
      maxPositionEmbeddings: 131072,
      hiddenSize: 2048,
      vocabSize: 128256,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 8192, numAttentionHeads: 32, tieEmbeddings: true },
    }),
  },
  {
    meta: { provider: 'meta', family: 'Llama 3.x', tags: ['chat', 'tool-use', 'multilingual', 'small-edge'], license: 'llama3.2', releaseDate: '2024-09' },
    spec: preset({
      id: 'meta-llama/Llama-3.2-3B-Instruct',
      name: 'Llama 3.2 3B',
      params: 3_212_749_824,
      numLayers: 28,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 3072,
      vocabSize: 128256,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 8192, numAttentionHeads: 24, tieEmbeddings: true },
    }),
  },
  {
    meta: { provider: 'meta', family: 'Llama 3.x', tags: ['chat', 'tool-use', 'multilingual', 'long-context'], license: 'llama3.1', releaseDate: '2024-07' },
    spec: preset({
      id: 'meta-llama/Llama-3.1-8B-Instruct',
      name: 'Llama 3.1 8B',
      params: 8_030_261_248,
      numLayers: 32,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 4096,
      vocabSize: 128256,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 14336, numAttentionHeads: 32, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'meta', family: 'Llama 3.x', tags: ['chat', 'tool-use', 'multilingual', 'long-context'], license: 'llama3.3', releaseDate: '2024-12' },
    spec: preset({
      id: 'meta-llama/Llama-3.3-70B-Instruct',
      name: 'Llama 3.3 70B',
      params: 70_553_706_496,
      numLayers: 80,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 8192,
      vocabSize: 128256,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 28672, numAttentionHeads: 64, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'meta', family: 'Llama 3.x', tags: ['chat', 'tool-use', 'multilingual', 'long-context'], license: 'llama3.1', releaseDate: '2024-07' },
    spec: preset({
      id: 'meta-llama/Llama-3.1-405B-Instruct',
      name: 'Llama 3.1 405B',
      params: 405_853_388_800,
      numLayers: 126,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 16384,
      vocabSize: 128256,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 53248, numAttentionHeads: 128, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'meta', family: 'Llama 4', tags: ['chat', 'vision', 'tool-use', 'multilingual', 'long-context'], license: 'llama4', releaseDate: '2025-04' },
    spec: preset({
      id: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
      name: 'Llama 4 Scout 17B',
      params: 108_641_793_536,
      numLayers: 48,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 10_485_760,
      hiddenSize: 5120,
      vocabSize: 202048,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 8192, numAttentionHeads: 40, tieEmbeddings: false },
      // sharedExperts: 1 is not a config.json field — Llama 4's MoE layers always run a
      // shared expert alongside the routed expert(s) per Meta's Llama 4 model card/blog.
      moe: { numExperts: 16, expertsPerToken: 1, sharedExperts: 1 },
      warnings: [VISION_WARNING],
    }),
  },

  // ---------------------------------------------------------------------------------------
  // Qwen — ungated.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'qwen', family: 'Qwen3', tags: ['chat', 'reasoning', 'multilingual', 'small-edge'], license: 'apache-2.0', releaseDate: '2025-04' },
    spec: preset({
      id: 'Qwen/Qwen3-4B',
      name: 'Qwen3-4B',
      params: 4_022_468_096,
      numLayers: 36,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 40960,
      hiddenSize: 2560,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 9728, numAttentionHeads: 32, tieEmbeddings: true },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen3', tags: ['chat', 'reasoning', 'multilingual', 'tool-use'], license: 'apache-2.0', releaseDate: '2025-04' },
    spec: preset({
      id: 'Qwen/Qwen3-8B',
      name: 'Qwen3-8B',
      params: 8_190_735_360,
      numLayers: 36,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 40960,
      hiddenSize: 4096,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 12288, numAttentionHeads: 32, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen3', tags: ['chat', 'reasoning', 'multilingual', 'tool-use'], license: 'apache-2.0', releaseDate: '2025-04' },
    spec: preset({
      id: 'Qwen/Qwen3-14B',
      name: 'Qwen3-14B',
      params: 14_768_307_200,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 40960,
      hiddenSize: 5120,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 17408, numAttentionHeads: 40, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen3', tags: ['chat', 'reasoning', 'multilingual', 'tool-use'], license: 'apache-2.0', releaseDate: '2025-04' },
    spec: preset({
      id: 'Qwen/Qwen3-32B',
      name: 'Qwen3-32B',
      params: 32_762_123_264,
      numLayers: 64,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 40960,
      hiddenSize: 5120,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 25600, numAttentionHeads: 64, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen3', tags: ['chat', 'reasoning', 'multilingual', 'tool-use'], license: 'apache-2.0', releaseDate: '2025-04' },
    spec: preset({
      id: 'Qwen/Qwen3-30B-A3B',
      name: 'Qwen3-30B-A3B',
      params: 30_532_122_624,
      numLayers: 48,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      maxPositionEmbeddings: 40960,
      hiddenSize: 2048,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 6144, moeIntermediateSize: 768, numAttentionHeads: 32, tieEmbeddings: false },
      moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen3', tags: ['chat', 'reasoning', 'multilingual', 'tool-use', 'long-context'], license: 'apache-2.0', releaseDate: '2025-04' },
    spec: preset({
      id: 'Qwen/Qwen3-235B-A22B',
      name: 'Qwen3-235B-A22B',
      params: 235_093_634_560,
      numLayers: 94,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      maxPositionEmbeddings: 40960,
      hiddenSize: 4096,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 12288, moeIntermediateSize: 1536, numAttentionHeads: 64, tieEmbeddings: false },
      moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen3-Coder', tags: ['coding', 'tool-use', 'long-context'], license: 'apache-2.0', releaseDate: '2025-07' },
    spec: preset({
      id: 'Qwen/Qwen3-Coder-30B-A3B-Instruct',
      name: 'Qwen3-Coder-30B-A3B',
      params: 30_532_122_624,
      numLayers: 48,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      maxPositionEmbeddings: 262144,
      hiddenSize: 2048,
      vocabSize: 151936,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 6144, moeIntermediateSize: 768, numAttentionHeads: 32, tieEmbeddings: false },
      moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 },
    }),
  },
  {
    meta: { provider: 'qwen', family: 'Qwen2.5-VL', tags: ['vision', 'chat', 'multilingual'], license: 'apache-2.0', releaseDate: '2025-01' },
    spec: preset({
      id: 'Qwen/Qwen2.5-VL-7B-Instruct',
      name: 'Qwen2.5-VL 7B',
      params: 8_292_166_656,
      numLayers: 28,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      maxPositionEmbeddings: 128000,
      hiddenSize: 3584,
      vocabSize: 152064,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 18944, numAttentionHeads: 28, tieEmbeddings: false },
      warnings: [VISION_WARNING],
    }),
  },

  // ---------------------------------------------------------------------------------------
  // Google — gated on the Hub (401 without a token). Values are the original hand-verified
  // presets, not re-fetched by the checker.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'google', family: 'Gemma 3', tags: ['chat', 'vision', 'multilingual', 'small-edge'], license: 'gemma', releaseDate: '2025-03' },
    spec: preset({
      id: 'google/gemma-3-4b-it',
      name: 'Gemma 3 4B',
      params: 4_300_079_472,
      numLayers: 34,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 256,
      slidingWindow: 1024,
      slidingLayers: 28,
      maxPositionEmbeddings: 131072,
      hiddenSize: 2560,
      vocabSize: 262208,
      nativeDtype: 'bf16',
      // tie_word_embeddings is not in config.json; Gemma 3 ties embeddings by default
      // (same as gemma-3-27b-it below).
      ffn: { intermediateSize: 10240, numAttentionHeads: 8, tieEmbeddings: true },
      warnings: ['parameter count includes the ~0.4B SigLIP vision tower'],
    }),
  },
  {
    meta: { provider: 'google', family: 'Gemma 3', tags: ['chat', 'vision', 'multilingual', 'long-context'], license: 'gemma', releaseDate: '2025-03' },
    spec: preset({
      id: 'google/gemma-3-12b-it',
      name: 'Gemma 3 12B',
      params: 12_187_325_040,
      numLayers: 48,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 256,
      slidingWindow: 1024,
      slidingLayers: 40,
      maxPositionEmbeddings: 131072,
      hiddenSize: 3840,
      vocabSize: 262208,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 15360, numAttentionHeads: 16, tieEmbeddings: true },
      warnings: ['parameter count includes the ~0.4B SigLIP vision tower'],
    }),
  },
  {
    meta: { provider: 'google', family: 'Gemma 3', tags: ['chat', 'vision', 'multilingual', 'long-context'], license: 'gemma', releaseDate: '2025-03' },
    spec: preset({
      id: 'google/gemma-3-27b-it',
      name: 'Gemma 3 27B',
      params: 27_432_406_640,
      numLayers: 62,
      attention: 'mha_gqa',
      numKvHeads: 16,
      headDim: 128,
      slidingWindow: 1024,
      slidingLayers: 51,
      maxPositionEmbeddings: 131072,
      hiddenSize: 5376,
      vocabSize: 262208,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 21504, numAttentionHeads: 32, tieEmbeddings: true },
      warnings: ['parameter count includes the ~0.4B SigLIP vision tower'],
    }),
  },

  // ---------------------------------------------------------------------------------------
  // Mistral — ungated.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'mistral', family: 'Mistral Small', tags: ['chat', 'vision', 'tool-use', 'multilingual'], license: 'apache-2.0', releaseDate: '2025-06' },
    spec: preset({
      id: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
      name: 'Mistral Small 3.2 24B',
      params: 24_011_361_280,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 5120,
      vocabSize: 131072,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 32768, numAttentionHeads: 32, tieEmbeddings: false },
      warnings: [VISION_WARNING],
    }),
  },
  {
    meta: { provider: 'mistral', family: 'Magistral', tags: ['reasoning', 'vision', 'multilingual'], license: 'apache-2.0', releaseDate: '2025-09' },
    spec: preset({
      id: 'mistralai/Magistral-Small-2509',
      name: 'Magistral Small 2509',
      params: 24_011_361_280,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 5120,
      vocabSize: 131072,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 32768, numAttentionHeads: 32, tieEmbeddings: false },
      warnings: [VISION_WARNING],
    }),
  },
  {
    meta: { provider: 'mistral', family: 'Devstral', tags: ['coding', 'tool-use', 'long-context'], license: 'apache-2.0', releaseDate: '2025-07' },
    spec: preset({
      id: 'mistralai/Devstral-Small-2507',
      name: 'Devstral Small 2507',
      params: 23_572_403_200,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 5120,
      vocabSize: 131072,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 32768, numAttentionHeads: 32, tieEmbeddings: false },
    }),
  },

  // ---------------------------------------------------------------------------------------
  // DeepSeek — ungated.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'deepseek', family: 'DeepSeek-V3', tags: ['chat', 'reasoning', 'coding', 'tool-use', 'long-context'], license: 'mit', releaseDate: '2025-08' },
    spec: preset({
      id: 'deepseek-ai/DeepSeek-V3.1',
      name: 'DeepSeek-V3.1',
      params: 684_531_386_000,
      numLayers: 61,
      attention: 'mla',
      numKvHeads: 128,
      headDim: 192,
      kvLoraRank: 512,
      qkRopeHeadDim: 64,
      maxPositionEmbeddings: 163840,
      hiddenSize: 7168,
      vocabSize: 129280,
      nativeDtype: 'fp8',
      ffn: {
        intermediateSize: 18432,
        moeIntermediateSize: 2048,
        firstKDense: 3,
        numAttentionHeads: 128,
        tieEmbeddings: false,
        qLoraRank: 1536,
        vHeadDim: 128,
      },
      moe: { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 },
    }),
  },
  {
    meta: { provider: 'deepseek', family: 'DeepSeek-R1', tags: ['reasoning', 'coding', 'long-context'], license: 'mit', releaseDate: '2025-01' },
    spec: preset({
      id: 'deepseek-ai/DeepSeek-R1',
      name: 'DeepSeek-R1',
      params: 684_489_845_504,
      numLayers: 61,
      attention: 'mla',
      numKvHeads: 128,
      headDim: 192,
      kvLoraRank: 512,
      qkRopeHeadDim: 64,
      maxPositionEmbeddings: 163840,
      hiddenSize: 7168,
      vocabSize: 129280,
      nativeDtype: 'fp8',
      ffn: {
        intermediateSize: 18432,
        moeIntermediateSize: 2048,
        firstKDense: 3,
        numAttentionHeads: 128,
        tieEmbeddings: false,
        qLoraRank: 1536,
        vHeadDim: 128,
      },
      moe: { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 },
    }),
  },
  {
    meta: { provider: 'deepseek', family: 'R1 Distill', tags: ['reasoning', 'coding'], license: 'apache-2.0', releaseDate: '2025-01', note: 'Qwen-32B base; DeepSeek distributes the distills under the base model’s license' },
    spec: preset({
      id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
      name: 'DeepSeek-R1-Distill-Qwen-32B',
      params: 32_763_876_352,
      numLayers: 64,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 5120,
      vocabSize: 152064,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 27648, numAttentionHeads: 40, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'deepseek', family: 'R1 Distill', tags: ['reasoning', 'coding', 'small-edge'], license: 'apache-2.0', releaseDate: '2025-01', note: 'Qwen-7B base; DeepSeek distributes the distills under the base model’s license' },
    spec: preset({
      id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
      name: 'DeepSeek-R1-Distill-Qwen-7B',
      params: 7_615_616_512,
      numLayers: 28,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 3584,
      vocabSize: 152064,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 18944, numAttentionHeads: 28, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'deepseek', family: 'R1 Distill', tags: ['reasoning', 'small-edge'], license: 'llama3.1', releaseDate: '2025-01', note: 'Llama-8B base; DeepSeek distributes the distills under the base model’s license' },
    spec: preset({
      id: 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
      name: 'DeepSeek-R1-Distill-Llama-8B',
      params: 8_030_261_248,
      numLayers: 32,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 4096,
      vocabSize: 128256,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 14336, numAttentionHeads: 32, tieEmbeddings: false },
    }),
  },

  // ---------------------------------------------------------------------------------------
  // OpenAI — ungated. Only two public sizes exist (120b, 20b); there is no third size to add.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'openai', family: 'gpt-oss', tags: ['chat', 'reasoning', 'tool-use', 'long-context'], license: 'apache-2.0', releaseDate: '2025-08' },
    spec: preset({
      id: 'openai/gpt-oss-120b',
      name: 'gpt-oss-120b',
      params: 116_829_156_672,
      numLayers: 36,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 64,
      slidingWindow: 128,
      slidingLayers: 18,
      maxPositionEmbeddings: 131072,
      hiddenSize: 2880,
      vocabSize: 201088,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 2880, numAttentionHeads: 64, tieEmbeddings: false },
      moe: { numExperts: 128, expertsPerToken: 4, sharedExperts: 0 },
      warnings: [MXFP4_NOTE],
    }),
  },
  {
    meta: { provider: 'openai', family: 'gpt-oss', tags: ['chat', 'reasoning', 'tool-use', 'small-edge'], license: 'apache-2.0', releaseDate: '2025-08' },
    spec: preset({
      id: 'openai/gpt-oss-20b',
      name: 'gpt-oss-20b',
      params: 20_914_757_184,
      numLayers: 24,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 64,
      slidingWindow: 128,
      slidingLayers: 12,
      maxPositionEmbeddings: 131072,
      hiddenSize: 2880,
      vocabSize: 201088,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 2880, numAttentionHeads: 64, tieEmbeddings: false },
      moe: { numExperts: 32, expertsPerToken: 4, sharedExperts: 0 },
      warnings: [MXFP4_NOTE],
    }),
  },

  // ---------------------------------------------------------------------------------------
  // Microsoft — ungated.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'microsoft', family: 'Phi-4', tags: ['chat', 'reasoning', 'small-edge', 'multilingual', 'tool-use'], license: 'mit', releaseDate: '2025-02' },
    spec: preset({
      id: 'microsoft/Phi-4-mini-instruct',
      name: 'Phi-4-mini-instruct',
      params: 3_836_021_760,
      numLayers: 32,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 3072,
      vocabSize: 200064,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 8192, numAttentionHeads: 24, tieEmbeddings: true },
    }),
  },
  {
    meta: { provider: 'microsoft', family: 'Phi-4', tags: ['chat', 'reasoning'], license: 'mit', releaseDate: '2024-12' },
    spec: preset({
      id: 'microsoft/phi-4',
      name: 'Phi-4',
      params: 14_659_507_200,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 10,
      headDim: 128,
      maxPositionEmbeddings: 16384,
      hiddenSize: 5120,
      vocabSize: 100352,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 17920, numAttentionHeads: 40, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'microsoft', family: 'Phi-4', tags: ['reasoning', 'coding'], license: 'mit', releaseDate: '2025-04' },
    spec: preset({
      id: 'microsoft/Phi-4-reasoning',
      name: 'Phi-4-reasoning',
      params: 14_659_507_200,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 10,
      headDim: 128,
      maxPositionEmbeddings: 32768,
      hiddenSize: 5120,
      vocabSize: 100352,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 17920, numAttentionHeads: 40, tieEmbeddings: false },
    }),
  },
  {
    meta: { provider: 'microsoft', family: 'Phi-4', tags: ['reasoning', 'coding'], license: 'mit', releaseDate: '2025-04' },
    spec: preset({
      id: 'microsoft/Phi-4-reasoning-plus',
      name: 'Phi-4-reasoning-plus',
      params: 14_659_507_200,
      numLayers: 40,
      attention: 'mha_gqa',
      numKvHeads: 10,
      headDim: 128,
      maxPositionEmbeddings: 32768,
      hiddenSize: 5120,
      vocabSize: 100352,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 17920, numAttentionHeads: 40, tieEmbeddings: false },
    }),
  },

  // ---------------------------------------------------------------------------------------
  // Moonshot — ungated.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'moonshot', family: 'Kimi K2', tags: ['chat', 'coding', 'tool-use', 'long-context'], license: 'modified-mit', releaseDate: '2025-07' },
    spec: preset({
      id: 'moonshotai/Kimi-K2-Instruct',
      name: 'Kimi K2 Instruct',
      params: 1_026_408_235_864,
      numLayers: 61,
      attention: 'mla',
      numKvHeads: 64,
      headDim: 192,
      kvLoraRank: 512,
      qkRopeHeadDim: 64,
      maxPositionEmbeddings: 131072,
      hiddenSize: 7168,
      vocabSize: 163840,
      nativeDtype: 'fp8',
      ffn: {
        intermediateSize: 18432,
        moeIntermediateSize: 2048,
        firstKDense: 1,
        numAttentionHeads: 64,
        tieEmbeddings: false,
        qLoraRank: 1536,
        vHeadDim: 128,
      },
      moe: { numExperts: 384, expertsPerToken: 8, sharedExperts: 1 },
    }),
  },

  // ---------------------------------------------------------------------------------------
  // Zhipu (zai-org on the Hub) — ungated.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'zhipu', family: 'GLM-4.5', tags: ['chat', 'reasoning', 'coding', 'tool-use', 'long-context'], license: 'mit', releaseDate: '2025-07' },
    spec: preset({
      id: 'zai-org/GLM-4.5',
      name: 'GLM-4.5',
      params: 358_337_791_296,
      numLayers: 92,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 5120,
      vocabSize: 151552,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 12288, moeIntermediateSize: 1536, firstKDense: 3, numAttentionHeads: 96, tieEmbeddings: false },
      moe: { numExperts: 160, expertsPerToken: 8, sharedExperts: 1 },
    }),
  },
  {
    meta: { provider: 'zhipu', family: 'GLM-4.5', tags: ['chat', 'reasoning', 'coding', 'tool-use'], license: 'mit', releaseDate: '2025-07' },
    spec: preset({
      id: 'zai-org/GLM-4.5-Air',
      name: 'GLM-4.5-Air',
      params: 110_468_824_832,
      numLayers: 46,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 4096,
      vocabSize: 151552,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 10944, moeIntermediateSize: 1408, firstKDense: 1, numAttentionHeads: 96, tieEmbeddings: false },
      moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 1 },
    }),
  },

  // ---------------------------------------------------------------------------------------
  // NVIDIA — mostly gated or non-standard. nvidia/Llama-3.1-Nemotron-70B-Instruct-HF is
  // gated (401) and nvidia/Llama-3.3-Nemotron-Super-49B-v1 uses a heterogeneous per-layer
  // ("puzzle" NAS) architecture with no single intermediate_size/head count to read, so both
  // are left out rather than guessing. Nemotron Nano 9B v2 has an ordinary config.
  // ---------------------------------------------------------------------------------------
  {
    meta: { provider: 'nvidia', family: 'Nemotron', tags: ['chat', 'reasoning', 'small-edge', 'long-context'], license: 'nvidia-open', releaseDate: '2025-08' },
    spec: preset({
      id: 'nvidia/NVIDIA-Nemotron-Nano-9B-v2',
      name: 'Nemotron Nano 9B v2',
      params: 8_888_227_328,
      numLayers: 56,
      attention: 'mha_gqa',
      numKvHeads: 8,
      headDim: 128,
      maxPositionEmbeddings: 131072,
      hiddenSize: 4480,
      vocabSize: 131072,
      nativeDtype: 'bf16',
      ffn: { intermediateSize: 15680, numAttentionHeads: 40, tieEmbeddings: false },
    }),
  },
];

/** All entries for one provider id, in catalog order. */
export function catalogByProvider(providerId: string): CatalogEntry[] {
  return MODEL_CATALOG.filter((e) => e.meta.provider === providerId);
}

/** A catalog entry by ModelSpec id or display name (case-insensitive), mirroring findModelPreset. */
export function findCatalogEntry(idOrName: string): CatalogEntry | undefined {
  const q = idOrName.trim().toLowerCase();
  return MODEL_CATALOG.find((e) => e.spec.id.toLowerCase() === q || e.spec.name.toLowerCase() === q);
}
