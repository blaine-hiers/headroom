import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { closestWeightQuant, effectiveBitsPerWeight, resolveWeights } from './fileWeights';
import { calculate } from './fit';
import type { CalcState } from './types';

const files = { bytes: 42.5e9, label: 'Q4_K_M GGUF', quant: 'q4_k_m' as const };

describe('resolveWeights', () => {
  it('uses the file bytes when the chosen quant is the files\' quant', () => {
    const spec = makeSpec({ params: 70e9, fileWeights: files });
    expect(resolveWeights(spec, 'q4_k_m', 70e9)).toEqual({ bytes: 42.5e9, activeBytes: 42.5e9, source: 'files' });
  });

  it('scales the file bytes by activeParams / params for MoE decode', () => {
    const spec = makeSpec({ params: 30e9, fileWeights: files });
    expect(resolveWeights(spec, 'q4_k_m', 3e9).activeBytes).toBeCloseTo(4.25e9, 0);
  });

  it('falls back to the estimate for any other quant, or without file weights', () => {
    const spec = makeSpec({ params: 70e9, fileWeights: files });
    expect(resolveWeights(spec, 'bf16', 70e9)).toEqual({ bytes: 140e9, activeBytes: 140e9, source: 'estimate' });
    expect(resolveWeights(makeSpec({ params: 70e9 }), 'q4_k_m', 70e9).source).toBe('estimate');
  });

  it('file bytes never apply to another quant; with unknown params decode reads all of them', () => {
    const spec = makeSpec({ params: 0, fileWeights: { bytes: 5e9, label: 'IQ2_M GGUF', quant: 'q2_k' } });
    expect(resolveWeights(spec, 'q2_k', 0)).toEqual({ bytes: 5e9, activeBytes: 5e9, source: 'files' });
    expect(resolveWeights(spec, 'bf16', 0).source).toBe('estimate');
  });
});

describe('closestWeightQuant / effectiveBitsPerWeight', () => {
  it('picks the nearest table quant by bits per weight', () => {
    expect(effectiveBitsPerWeight(4.6e9, 7.6e9)).toBeCloseTo(4.84, 2);
    expect(closestWeightQuant(4.84)).toBe('q4_k_m');
    expect(closestWeightQuant(15.9)).toBe('bf16');
    expect(closestWeightQuant(0)).toBeUndefined();
  });
});

describe('calculate with file weights', () => {
  const state: CalcState = {
    model: makeSpec({ fileWeights: files }),
    quant: { weight: 'q4_k_m', kv: 'fp16' },
    hardware: { gpuName: 'H100', gpuCount: 1, vramGB: 80, bandwidthGBs: 3350, reservePct: 5, overheadGB: 1 },
    workload: { contextTokens: 8192, concurrentUsers: 1 },
  };

  it('puts the exact bytes into the total and reports the source', () => {
    const r = calculate(state);
    expect(r.weightSource).toBe('files');
    expect(r.weightBytes).toBe(42.5e9);
    expect(r.activeWeightBytes).toBe(42.5e9); // dense: all of it
    expect(r.totalBytes).toBe(42.5e9 + 1e9 + r.kvBytesAllUsers);
    const est = calculate({ ...state, quant: { ...state.quant, weight: 'q8_0' } });
    expect(est.weightSource).toBe('estimate');
  });
});
