import type { ModelSpec, MoeSpec } from '../types';
import { activeParams } from '../weights';

// Values checked 2026-09-23 against each repo's config.json and the Hub API's
// safetensors.total (gated Llama/Gemma: API param count + published config values).

type PresetInput = Omit<ModelSpec, 'activeParams' | 'source' | 'warnings'> & { warnings?: string[] };

function preset(p: PresetInput): ModelSpec {
  const moe: MoeSpec | undefined = p.moe;
  return {
    slidingLayers: 0,
    ...p,
    activeParams: activeParams(p.params, moe),
    source: 'preset',
    warnings: p.warnings ?? [],
  };
}

const MOE_NOTE =
  'MoE: all experts must be resident; active params are an estimate that slightly understates (attention and embeddings are always active)';

export const MODEL_PRESETS: ModelSpec[] = [
  preset({
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
  }),
  preset({
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
  }),
  preset({
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
  }),
  preset({
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
  }),
  preset({
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
    moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 },
    warnings: [MOE_NOTE],
  }),
  preset({
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
    warnings: [
      'sliding-window attention on 51 of 62 layers (window 1024 tokens): KV for those layers stops growing past the window; some runtimes ignore this and allocate full-context KV',
      'param count includes the ~0.4B SigLIP vision tower',
    ],
  }),
  preset({
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
    moe: { numExperts: 128, expertsPerToken: 4, sharedExperts: 0 },
    warnings: [
      'sliding-window attention on 18 of 36 layers (window 128 tokens): KV for those layers stops growing past the window; some runtimes ignore this and allocate full-context KV',
      MOE_NOTE,
      'weights ship as MXFP4 (~4.25 bits/weight for the experts); pick a 4-bit weight quant to match the download size',
    ],
  }),
  preset({
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
    moe: { numExperts: 32, expertsPerToken: 4, sharedExperts: 0 },
    warnings: [
      'sliding-window attention on 12 of 24 layers (window 128 tokens): KV for those layers stops growing past the window; some runtimes ignore this and allocate full-context KV',
      MOE_NOTE,
      'weights ship as MXFP4 (~4.25 bits/weight for the experts); pick a 4-bit weight quant to match the download size',
    ],
  }),
  preset({
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
    moe: { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 },
    warnings: [
      'MLA: KV cache is a single compressed latent per token (kv_lora_rank + qk_rope_head_dim), no separate K and V',
      MOE_NOTE,
      'native weights are FP8; the quant table starts at fp8 for this model',
    ],
  }),
  preset({
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
  }),
];

export function findModelPreset(idOrName: string): ModelSpec | undefined {
  const q = idOrName.trim().toLowerCase();
  return MODEL_PRESETS.find((m) => m.id.toLowerCase() === q || m.name.toLowerCase() === q);
}
