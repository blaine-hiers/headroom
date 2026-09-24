import { describe, expect, it } from 'vitest';
import { CUSTOM_GPU_NAME, encodeState, findGpuPreset, findModelPreset } from '../lib';
import type { CalcState } from '../lib';
import { calculate } from '../lib';
import { clampModel, defaultState, initialState, MAX_DIM, MAX_FILE_BYTES, reducer } from './state';

function preset(name: string) {
  const m = findModelPreset(name);
  if (!m) throw new Error(`missing preset ${name}`);
  return m;
}

describe('reducer: file weights', () => {
  const gguf: CalcState = {
    ...defaultState,
    model: { ...defaultState.model, source: 'hf', fileWeights: { bytes: 42.5e9, label: 'Q4_K_M GGUF', quant: 'q4_k_m' } },
  };

  it('loading a spec with file weights pre-selects their quant', () => {
    const s = reducer(defaultState, { type: 'loadModel', spec: gguf.model });
    expect(s.quant.weight).toBe('q4_k_m');
    expect(calculate(s).weightSource).toBe('files');
  });

  it('an Advanced edit to params or the architecture drops the file weights', () => {
    const loaded = reducer(defaultState, { type: 'loadModel', spec: gguf.model });
    for (const patch of [{ params: 8e9 }, { numLayers: 40 }, { headDim: 64 }, { moe: undefined }]) {
      const s = reducer(loaded, { type: 'editModel', patch });
      expect(s.model.fileWeights).toBeUndefined();
      expect(calculate(s).weightSource).toBe('estimate');
    }
    // The native dtype says nothing about the files: they stay.
    expect(reducer(loaded, { type: 'editModel', patch: { nativeDtype: 'fp16' } }).model.fileWeights).toBeDefined();
  });

  it('a link with huge file bytes is capped like params', () => {
    const s = initialState(`?${encodeState({ ...gguf, model: { ...gguf.model, fileWeights: { bytes: 1e30, label: 'x', quant: 'q4_k_m' } } })}`);
    expect(s.model.fileWeights?.bytes).toBe(MAX_FILE_BYTES);
    expect(clampModel({ ...gguf.model, fileWeights: { bytes: 5e9, label: 'x', quant: 'q4_k_m' } }).fileWeights?.bytes).toBe(5e9);
  });
});

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

describe('reducer: hardware cloud cost', () => {
  it('switching to a GPU preset pre-fills its cloud cost', () => {
    const s = reducer(defaultState, { type: 'hardware', patch: { gpuName: 'H100 SXM' } });
    expect(s.hardware.usdPerHour).toBe(findGpuPreset('H100 SXM')?.usdPerHour);
  });

  it('switching to a preset with no listed price clears any previous one', () => {
    const withPrice = reducer(defaultState, { type: 'hardware', patch: { gpuName: 'H100 SXM' } });
    const s = reducer(withPrice, { type: 'hardware', patch: { gpuName: 'RTX 4090' } });
    expect(s.hardware.usdPerHour).toBeUndefined();
  });

  it('an edited price is kept until the GPU preset changes', () => {
    const s = reducer(defaultState, { type: 'hardware', patch: { usdPerHour: 1.23 } });
    expect(s.hardware.usdPerHour).toBe(1.23);
  });
});

describe('initialState: a cleared cloud cost price stays cleared', () => {
  it('does not get clamped back up to the 0.01 minimum on reload', () => {
    // H100 SXM has a preset price; usdPerHour: undefined here is a deliberate clear, which
    // urlState.ts encodes as an explicit sentinel (not an absent key). initialState's clamp
    // only touches a defined usdPerHour, so the cleared value must stay undefined, not 0.01.
    const s = { ...defaultState, hardware: { ...defaultState.hardware, gpuName: 'H100 SXM', usdPerHour: undefined } };
    const decoded = initialState(`?${encodeState(s)}`);
    expect(decoded.hardware.usdPerHour).toBeUndefined();
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
