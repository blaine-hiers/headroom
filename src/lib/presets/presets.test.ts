import { describe, expect, it } from 'vitest';
import { kvBytesForContext, kvBytesPerToken } from '../kvcache';
import { deriveWarnings } from '../hf';
import { activeParamsDetailed } from '../weights';
import { CUSTOM_GPU_NAME, GPU_PRESETS, findGpuPreset } from './gpus';
import { MODEL_PRESETS, findModelPreset } from './models';

describe('GPU presets', () => {
  it('has the required entries with the right numbers', () => {
    const expected: Array<[string, number, number]> = [
      ['RTX 2070', 8, 448], ['RTX 3060', 12, 360], ['RTX 3090', 24, 936], ['RTX 4070 Ti Super', 16, 672],
      ['RTX 4080', 16, 717], ['RTX 4090', 24, 1008], ['RTX 5080', 16, 960], ['RTX 5090', 32, 1792],
      ['RTX 6000 Ada', 48, 960], ['RTX PRO 6000 Blackwell', 96, 1792], ['L4', 24, 300], ['L40S', 48, 864],
      ['A10', 24, 600], ['A100 40GB', 40, 1555], ['A100 80GB', 80, 2039], ['H100 PCIe', 80, 2000],
      ['H100 SXM', 80, 3350], ['H200', 141, 4800], ['B200', 192, 8000], ['AMD MI300X', 192, 5300],
      ['Radeon RX 7900 XTX', 24, 960], ['Apple M4 Max', 128, 546], ['Apple M3 Ultra', 512, 819],
      ['Apple M2 Ultra', 192, 800], ['NVIDIA DGX Spark', 128, 273],
    ];
    for (const [name, vram, bw] of expected) {
      expect(findGpuPreset(name), name).toMatchObject({ vramGB: vram, bandwidthGBs: bw });
    }
    expect(findGpuPreset(CUSTOM_GPU_NAME)).toBeDefined();
  });

  it('names are unique and numbers positive', () => {
    expect(new Set(GPU_PRESETS.map((g) => g.name)).size).toBe(GPU_PRESETS.length);
    for (const g of GPU_PRESETS) {
      expect(g.vramGB).toBeGreaterThan(0);
      expect(g.bandwidthGBs).toBeGreaterThan(0);
    }
  });

  it('every preset has a sane dense BF16 TFLOPS figure', () => {
    for (const g of GPU_PRESETS) {
      expect(Number.isFinite(g.tflopsBf16), g.name).toBe(true);
      // Bracket: below the slowest listed card (RTX 2070, 14.9) and above the fastest (B200, 2250).
      expect(g.tflopsBf16, g.name).toBeGreaterThan(10);
      expect(g.tflopsBf16, g.name).toBeLessThan(3000);
    }
  });
});

describe('model presets', () => {
  it('are well-formed', () => {
    expect(new Set(MODEL_PRESETS.map((m) => m.id)).size).toBe(MODEL_PRESETS.length);
    for (const m of MODEL_PRESETS) {
      expect(m.source).toBe('preset');
      expect(m.id).toMatch(/^[\w.-]+\/[\w.-]+$/);
      expect(m.activeParams).toBe(activeParamsDetailed(m).active);
      expect(m.ffn, m.name).toBeDefined();
      expect(m.warnings.slice(0, deriveWarnings(m).length)).toEqual(deriveWarnings(m));
      expect(Number.isFinite(kvBytesPerToken(m, 'fp16'))).toBe(true);
      expect(kvBytesPerToken(m, 'fp16')).toBeGreaterThan(0);
    }
  });

  it('carry the spec values', () => {
    expect(findModelPreset('Llama 3.1 8B')).toMatchObject({ numLayers: 32, numKvHeads: 8, headDim: 128, hiddenSize: 4096, maxPositionEmbeddings: 131072 });
    expect(findModelPreset('Llama 3.3 70B')).toMatchObject({ numLayers: 80, numKvHeads: 8, headDim: 128, hiddenSize: 8192 });
    expect(findModelPreset('Qwen/Qwen3-32B')).toMatchObject({ numLayers: 64, numKvHeads: 8, hiddenSize: 5120, maxPositionEmbeddings: 40960 });
    expect(findModelPreset('Qwen3-30B-A3B')?.moe).toEqual({ numExperts: 128, expertsPerToken: 8, sharedExperts: 0 });
    expect(findModelPreset('Gemma 3 27B')).toMatchObject({ numLayers: 62, slidingLayers: 51, slidingWindow: 1024, numKvHeads: 16 });
    expect(findModelPreset('gpt-oss-120b')).toMatchObject({ numLayers: 36, slidingLayers: 18, slidingWindow: 128, headDim: 64 });
    expect(findModelPreset('gpt-oss-20b')).toMatchObject({ numLayers: 24, slidingLayers: 12 });
    expect(findModelPreset('DeepSeek-V3.1')).toMatchObject({ attention: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, nativeDtype: 'fp8', maxPositionEmbeddings: 163840 });
    expect(findModelPreset('Mistral Small 3.2 24B')).toMatchObject({ numLayers: 40, numKvHeads: 8, headDim: 128 });
    expect(findModelPreset('nope')).toBeUndefined();
  });

  it('golden: Llama 3.3 70B preset at 128K = 40 GiB of KV', () => {
    const m = findModelPreset('Llama 3.3 70B');
    if (!m) throw new Error('missing');
    expect(kvBytesForContext(m, 131072, 'fp16')).toBe(40 * 1024 ** 3);
  });
});
