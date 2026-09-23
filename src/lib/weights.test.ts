import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import deepseekRaw from './__fixtures__/deepseek-v3.json?raw';
import glmAirRaw from './__fixtures__/glm-4.5-air.json?raw';
import gptOss120Raw from './__fixtures__/gpt-oss-120b.json?raw';
import gptOss20Raw from './__fixtures__/gpt-oss-20b.json?raw';
import kimiRaw from './__fixtures__/kimi-k2.json?raw';
import qwen235Raw from './__fixtures__/qwen3-235b-a22b.json?raw';
import qwen30Raw from './__fixtures__/qwen3-30b-a3b.json?raw';
import { parseConfig } from './hf';
import { findModelPreset } from './presets/models';
import { activeParams, activeParamsDetailed, weightBytes } from './weights';

describe('weightBytes', () => {
  it('is params × bits / 8', () => {
    expect(weightBytes(8e9, 'bf16')).toBe(16e9);
    expect(weightBytes(8e9, 'fp32')).toBe(32e9);
    expect(weightBytes(8e9, 'fp8')).toBe(8e9);
    expect(weightBytes(8e9, 'q4_k_m')).toBeCloseTo(4.85e9, 0);
    expect(weightBytes(8e9, 'q8_0')).toBe(8.5e9);
  });
});

describe('activeParams', () => {
  it('dense = params', () => {
    expect(activeParams(70e9)).toBe(70e9);
    expect(activeParams(70e9, { numExperts: 1, expertsPerToken: 1, sharedExperts: 0 })).toBe(70e9);
  });

  it('Qwen3-30B-A3B: 128 experts, 8 per token', () => {
    expect(activeParams(30.5e9, { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 })).toBeCloseTo(
      (30.5e9 * 8) / 128,
      0,
    );
  });

  it('DeepSeek-V3: 256 routed + 1 shared, 8 per token', () => {
    expect(activeParams(685e9, { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 })).toBeCloseTo(
      (685e9 * 9) / 257,
      0,
    );
  });

  it('never exceeds params', () => {
    expect(activeParams(10e9, { numExperts: 4, expertsPerToken: 8, sharedExperts: 0 })).toBe(10e9);
  });
});

describe('activeParamsDetailed', () => {
  // Published active-parameter figures; the structural estimate must land within ±10%.
  const golden: Array<[string, string, number]> = [
    ['Qwen3-30B-A3B', qwen30Raw, 3.3e9],
    ['Qwen3-235B-A22B', qwen235Raw, 22e9],
    ['gpt-oss-20b', gptOss20Raw, 3.6e9],
    ['gpt-oss-120b', gptOss120Raw, 5.1e9],
    ['DeepSeek-V3', deepseekRaw, 37e9],
    ['GLM-4.5-Air', glmAirRaw, 12e9],
    ['Kimi-K2', kimiRaw, 32e9],
  ];
  for (const [name, raw, published] of golden) {
    it(`golden: ${name} ≈ ${published / 1e9}B active (structural, within ±10%)`, () => {
      const spec = parseConfig(JSON.parse(raw), undefined, `org/${name}`);
      const { active, method } = activeParamsDetailed(spec);
      expect(method).toBe('structural');
      expect(Math.abs(active / published - 1)).toBeLessThan(0.1);
      expect(spec.activeParams).toBe(active);
    });
  }

  it('MoE presets use the structural estimate', () => {
    for (const name of ['Qwen3-30B-A3B', 'gpt-oss-20b', 'gpt-oss-120b', 'DeepSeek-V3.1']) {
      const m = findModelPreset(name);
      if (!m) throw new Error(name);
      expect(activeParamsDetailed(m).method, name).toBe('structural');
    }
  });

  it('dense models use every param', () => {
    expect(activeParamsDetailed(makeSpec())).toEqual({ active: 70.6e9, method: 'dense' });
    expect(activeParamsDetailed(makeSpec({ moe: { numExperts: 1, expertsPerToken: 1, sharedExperts: 0 } })).method).toBe('dense');
  });

  it('MoE without layer shapes falls back to the ratio', () => {
    const spec = makeSpec({ params: 30.5e9, moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 } });
    expect(activeParamsDetailed(spec)).toEqual({ active: (30.5e9 * 8) / 128, method: 'ratio' });
  });

  it('the structural estimate is well above the ratio for fine-grained MoE', () => {
    const m = findModelPreset('Qwen3-30B-A3B');
    if (!m) throw new Error('missing');
    expect(activeParamsDetailed(m).active / activeParams(m.params, m.moe)).toBeGreaterThan(1.25);
  });

  it('never exceeds the total params when they are known', () => {
    const spec = makeSpec({
      params: 1e9,
      hiddenSize: 4096,
      moe: { numExperts: 8, expertsPerToken: 2, sharedExperts: 0 },
      ffn: { intermediateSize: 14336, numAttentionHeads: 32, tieEmbeddings: false },
    });
    expect(activeParamsDetailed(spec)).toEqual({ active: 1e9, method: 'structural' });
  });

  it('MLA without v_head_dim falls back to the GQA attention formula', () => {
    const base = parseConfig(JSON.parse(deepseekRaw), undefined, 'a/b');
    if (!base.ffn) throw new Error('ffn not parsed');
    const noV = { ...base, ffn: { ...base.ffn, vHeadDim: undefined } };
    expect(activeParamsDetailed(noV).method).toBe('structural');
    expect(activeParamsDetailed(noV).active).not.toBe(activeParamsDetailed(base).active);
  });
});
