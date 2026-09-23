import { describe, expect, it } from 'vitest';
import { CUSTOM_GPU_NAME, encodeState, findModelPreset } from '../lib';
import type { CalcState } from '../lib';
import { clampModel, defaultState, initialState, MAX_DIM, reducer } from './state';

function preset(name: string) {
  const m = findModelPreset(name);
  if (!m) throw new Error(`missing preset ${name}`);
  return m;
}

describe('reducer: warnings follow Advanced edits', () => {
  const gptOss: CalcState = { ...defaultState, model: preset('gpt-oss-120b') };

  it('turning MoE off drops the MoE note and makes active params dense', () => {
    const s = reducer(gptOss, { type: 'editModel', patch: { moe: undefined } });
    expect(s.model.warnings.some((w) => w.startsWith('MoE'))).toBe(false);
    expect(s.model.activeParams).toBe(s.model.params);
    // config-only notes survive
    expect(s.model.warnings.some((w) => w.includes('MXFP4'))).toBe(true);
  });

  it('clearing the sliding window drops the sliding-window note', () => {
    const s = reducer(gptOss, { type: 'editModel', patch: { slidingWindow: 0 } });
    expect(s.model.warnings.some((w) => w.startsWith('sliding-window'))).toBe(false);
  });

  it('turning MoE on for a dense model adds the note once', () => {
    const s = reducer(defaultState, { type: 'editModel', patch: { moe: { numExperts: 8, expertsPerToken: 2, sharedExperts: 0 } } });
    expect(s.model.warnings.filter((w) => w.startsWith('MoE'))).toHaveLength(1);
    const again = reducer(s, { type: 'editModel', patch: { params: 50e9 } });
    expect(again.model.warnings.filter((w) => w.startsWith('MoE'))).toHaveLength(1);
  });

  it('switching native dtype to FP8 adds the FP8 note, and back removes it', () => {
    const fp8 = reducer(defaultState, { type: 'editModel', patch: { nativeDtype: 'fp8' } });
    expect(fp8.model.warnings).toContain('native FP8 weights: FP8 pre-selected; BF16 would double the weight size');
    const back = reducer(fp8, { type: 'editModel', patch: { nativeDtype: 'bf16' } });
    expect(back.model.warnings.some((w) => w.includes('FP8'))).toBe(false);
  });
});

describe('initialState from a URL', () => {
  it('clamps out-of-range model fields', () => {
    const model = {
      ...preset('Qwen3-30B-A3B'),
      numLayers: 5000,
      numKvHeads: -3,
      headDim: 1e12,
      maxPositionEmbeddings: 10,
      slidingWindow: -1,
      slidingLayers: 99999,
      params: -5,
    };
    const s = initialState(`?${encodeState({ ...defaultState, model })}`);
    expect(s.model.numLayers).toBe(1000);
    expect(s.model.numKvHeads).toBe(0);
    expect(s.model.headDim).toBe(MAX_DIM);
    expect(s.model.maxPositionEmbeddings).toBe(256);
    expect(s.model.slidingWindow).toBe(0);
    expect(s.model.slidingLayers).toBe(1000);
    expect(s.model.params).toBe(0);
  });

  it('recomputes active params from the shared shapes (the URL value is not trusted)', () => {
    const m = preset('Qwen3-30B-A3B');
    const s = initialState(`?${encodeState({ ...defaultState, model: { ...m, activeParams: 1 } })}`);
    expect(s.model.activeParams).toBe(m.activeParams);
  });

  it('clampModel leaves a preset unchanged', () => {
    for (const name of ['Llama 3.3 70B', 'DeepSeek-V3.1', 'gpt-oss-20b', 'Gemma 3 27B']) {
      const m = preset(name);
      expect(clampModel(m), name).toEqual(m);
    }
  });

  it('an unknown GPU name becomes Custom and keeps the link VRAM and bandwidth', () => {
    const hw = { ...defaultState.hardware, gpuName: 'Imaginary 9000', vramGB: 37, bandwidthGBs: 1234 };
    const s = initialState(`?${encodeState({ ...defaultState, hardware: hw })}`);
    expect(s.hardware).toMatchObject({ gpuName: CUSTOM_GPU_NAME, vramGB: 37, bandwidthGBs: 1234 });
  });

  it('a known GPU name is kept', () => {
    const s = initialState(`?${encodeState(defaultState)}`);
    expect(s.hardware.gpuName).toBe(defaultState.hardware.gpuName);
  });
});
