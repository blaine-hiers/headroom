import type { ModelSpec } from '../types';

/** Test helper: a dense GQA ModelSpec with Llama 3 70B defaults, overridable. */
export function makeSpec(overrides: Partial<ModelSpec> = {}): ModelSpec {
  return {
    id: 'test/llama-3-70b',
    name: 'Llama 3 70B',
    params: 70.6e9,
    activeParams: 70.6e9,
    numLayers: 80,
    attention: 'mha_gqa',
    numKvHeads: 8,
    headDim: 128,
    slidingLayers: 0,
    maxPositionEmbeddings: 131072,
    hiddenSize: 8192,
    vocabSize: 128256,
    nativeDtype: 'bf16',
    source: 'manual',
    warnings: [],
    ...overrides,
  };
}
