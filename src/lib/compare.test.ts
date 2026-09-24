import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import type { CalcState } from './types';
import { bestColumnIndex, decodeCompareColumns, encodeCompareState } from './compare';
import { decodeState, encodeState } from './urlState';

const base: CalcState = {
  model: makeSpec(),
  quant: { weight: 'bf16', kv: 'fp16' },
  hardware: { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 165.0, reservePct: 5, overheadGB: 1 },
  workload: { contextTokens: 8192, concurrentUsers: 1 },
  runtime: 'generic',
};

const columnB: CalcState = {
  ...base,
  model: makeSpec({ id: 'org/model-b', name: 'Model B' }),
  workload: { contextTokens: 32768, concurrentUsers: 4 },
  runtime: 'vllm',
};

const columnC: CalcState = {
  ...base,
  hardware: { ...base.hardware, gpuName: 'H100 SXM', gpuCount: 2, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5, reservePct: 5, overheadGB: 1 },
  quant: { weight: 'q4_k_m', kv: 'fp8' },
};

describe('compare', () => {
  it('a single-column state encodes exactly like the plain encodeState (old URLs unaffected)', () => {
    expect(encodeCompareState([base])).toBe(encodeState(base));
  });

  it('round-trips a 2-column state', () => {
    const qs = encodeCompareState([base, columnB]);
    expect(decodeState(qs, base)).toEqual(base);
    expect(decodeCompareColumns(qs)).toEqual([columnB]);
    expect(decodeCompareColumns(`?${qs}`)).toEqual([columnB]);
  });

  it('round-trips a 3-column state', () => {
    const qs = encodeCompareState([base, columnB, columnC]);
    expect(decodeState(qs, base)).toEqual(base);
    expect(decodeCompareColumns(qs)).toEqual([columnB, columnC]);
  });

  it('stops at the first missing extra column, never producing an A/_/C gap', () => {
    // Hand-build a URL with only c3 set (no c2): c3 must be ignored, not read as column 2.
    const qs = `${encodeState(base)}&c3=${encodeURIComponent(encodeState(columnC))}`;
    expect(decodeCompareColumns(qs)).toEqual([]);
  });

  it('drops a corrupt extra column instead of throwing or emitting a partial state', () => {
    const qs = `${encodeState(base)}&c2=not-a-valid-state`;
    expect(decodeCompareColumns(qs)).toEqual([]);
  });

  it('ignores unknown c2/c3 keys when decoding the primary column (forward compatibility)', () => {
    const qs = encodeCompareState([base, columnB]);
    expect(decodeState(qs, base)).toEqual(base);
  });
});

describe('bestColumnIndex', () => {
  it('picks the single lowest value when lower is better', () => {
    expect(bestColumnIndex([30, 10, 20], false)).toBe(1);
  });

  it('picks the single highest value when higher is better', () => {
    expect(bestColumnIndex([30, 10, 20], true)).toBe(0);
  });

  it('returns -1 on a tie, so nothing is highlighted', () => {
    expect(bestColumnIndex([10, 10, 20], false)).toBe(-1);
  });

  it('treats Infinity as the best "higher is better" value', () => {
    expect(bestColumnIndex([100, Number.POSITIVE_INFINITY, 50], true)).toBe(1);
  });

  it('returns -1 when every value is non-finite', () => {
    expect(bestColumnIndex([NaN, NaN], true)).toBe(-1);
  });
});
