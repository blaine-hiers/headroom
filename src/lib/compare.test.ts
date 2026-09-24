import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import type { CalcState } from './types';
import { bestColumnIndex, COMPARE_ROWS, compareRowWinner, decodeCompareColumns, encodeCompareState } from './compare';
import { calculate } from './fit';
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

  it('skips an ineligible column even when it holds the best raw value (a non-fitting column cannot win tok/s)', () => {
    // Column 1 (index 1) has the highest throughput but doesn't fit, so it must not win.
    expect(bestColumnIndex([100, 500, 200], true, [true, false, true])).toBe(2);
  });

  it('returns -1 when every column is ineligible', () => {
    expect(bestColumnIndex([100, 500, 200], true, [false, false, false])).toBe(-1);
  });

  it('with no eligible column ruled out, behaves exactly like the 2-arg form', () => {
    expect(bestColumnIndex([100, 500, 200], true, [true, true, true])).toBe(1);
  });
});

describe('COMPARE_ROWS', () => {
  it('only excludes non-fitting columns from winning the rows whose number assumes the model is resident', () => {
    const excluded = COMPARE_ROWS.filter((r) => r.excludeUnfitFromBest).map((r) => r.key);
    expect(excluded.sort()).toEqual(['maxContext', 'maxUsers', 'tokS'].sort());
  });
});

describe('compareRowWinner (#20: offloaded columns that run can win)', () => {
  const q4 = { weight: 'q4_k_m', kv: 'fp16' } as const;
  const offloaded: CalcState = {
    ...base,
    quant: q4,
    hardware: { ...base.hardware, offload: { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 } },
    workload: { contextTokens: 2048, concurrentUsers: 1 },
  };
  const noOffload: CalcState = { ...offloaded, hardware: base.hardware };
  const row = (key: string) => COMPARE_ROWS.find((r) => r.key === key)!;

  it('an offloaded column that runs is eligible for best tok/s and max users; the same config without offload is not', () => {
    const a = calculate(offloaded);
    const b = calculate(noOffload);
    expect(a.fits).toBe(false);
    expect(a.runs).toBe(true);
    expect(b.runs).toBe(false);
    expect(compareRowWinner(row('tokS'), [a, b])).toBe(0);
    expect(compareRowWinner(row('maxUsers'), [a, b])).toBe(0);
    // Headroom compares the runs-aware figure, so the column that runs wins that row too.
    expect(compareRowWinner(row('headroom'), [a, b])).toBe(0);
  });

  it('an offloaded split that does not run stays ineligible', () => {
    const broken = calculate({ ...offloaded, hardware: { ...base.hardware, offload: { enabled: true, systemRamGB: 1, ramBandwidthGBs: 50 } } });
    expect(broken.runs).toBe(false);
    expect(compareRowWinner(row('maxUsers'), [broken, calculate(noOffload)])).toBe(-1);
  });
});
