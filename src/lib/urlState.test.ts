import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { MODEL_PRESETS } from './presets/models';
import type { CalcState } from './types';
import { decodeState, encodeState } from './urlState';
import { activeParamsDetailed } from './weights';

const fallback: CalcState = {
  model: makeSpec(),
  quant: { weight: 'bf16', kv: 'fp16' },
  hardware: { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 82.6, reservePct: 5, overheadGB: 1 },
  workload: { contextTokens: 8192, concurrentUsers: 1 },
};

describe('urlState', () => {
  it('round-trips every preset exactly', () => {
    for (const model of MODEL_PRESETS) {
      const s: CalcState = {
        model,
        quant: { weight: 'q4_k_m', kv: 'fp8' },
        hardware: { gpuName: 'H100 SXM', gpuCount: 8, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5, reservePct: 7.5, overheadGB: 1.25 },
        workload: { contextTokens: 32768, concurrentUsers: 16 },
      };
      const qs = encodeState(s);
      expect(decodeState(qs, fallback)).toEqual(s);
      expect(decodeState(`?${qs}`, fallback)).toEqual(s);
    }
  });

  it('round-trips awkward strings and float params', () => {
    const s: CalcState = {
      ...fallback,
      model: makeSpec({ id: 'org/näme & co', name: 'a=b?c#d', params: 70_553_706_496.5, warnings: ['x, y', 'z&w'], source: 'manual' }),
    };
    expect(decodeState(encodeState(s), fallback)).toEqual(s);
  });

  it('round-trips the FFN shapes, so a shared MoE link keeps its structural active params', () => {
    const s: CalcState = {
      ...fallback,
      model: makeSpec({
        moe: { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 },
        ffn: { intermediateSize: 18432, moeIntermediateSize: 2048, firstKDense: 3, numAttentionHeads: 128, tieEmbeddings: false, qLoraRank: 1536 },
      }),
    };
    const decoded = decodeState(encodeState(s), fallback);
    expect(decoded).toEqual(s);
    expect(activeParamsDetailed(decoded.model)).toEqual(activeParamsDetailed(s.model));
    expect(activeParamsDetailed(decoded.model).method).toBe('structural');
  });

  it('uses compact keys', () => {
    const qs = encodeState(fallback);
    expect(qs).toContain('wq=bf16');
    expect(qs).toContain('c=8192');
    expect(qs.length).toBeLessThan(400);
  });

  it('an old link without tflopsBf16 (issue #11) still decodes, defaulting the new field', () => {
    const qs = encodeState(fallback).replace(/&?tf=[^&]*/, '');
    expect(qs).not.toContain('tf=');
    const decoded = decodeState(qs, fallback);
    expect(decoded).not.toBe(fallback);
    expect(decoded.hardware.tflopsBf16).toBe(100);
    expect(decoded).toEqual({ ...fallback, hardware: { ...fallback.hardware, tflopsBf16: 100 } });
  });

  it('bad input → fallback', () => {
    expect(decodeState('', fallback)).toBe(fallback);
    expect(decodeState('garbage', fallback)).toBe(fallback);
    const good = encodeState(fallback);
    expect(decodeState(good.replace('wq=bf16', 'wq=nope'), fallback)).toBe(fallback);
    expect(decodeState(good.replace('c=8192', 'c=abc'), fallback)).toBe(fallback);
    expect(decodeState(good.replace('c=8192', 'c='), fallback)).toBe(fallback);
    expect(decodeState(good.replace('at=mha_gqa', 'at=weird'), fallback)).toBe(fallback);
    expect(decodeState(good.replace(/&u=\d+/, ''), fallback)).toBe(fallback);
    expect(decodeState(`${good}&moe=1,2`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=1,2,1`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=1,2,yes,,,,`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=,2,1,,,,`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=1,2,0,x,,,`, fallback)).toBe(fallback);
  });
});
