import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { calculate } from './fit';
import {
  evaluateHardwareSizingRow,
  rankHardwareSizingRows,
  scalingStrip,
  SCALING_STRIP_USER_COUNTS,
  sizeHardware,
} from './hardwareSizing';
import type { HardwareSizingOptions, HardwareSizingRow } from './hardwareSizing';
import { findGpuPreset } from './presets/gpus';
import type { CalcState, HardwareSpec } from './types';

const baseOptions: HardwareSizingOptions = {
  quant: { weight: 'bf16', kv: 'fp16' },
  runtime: 'generic',
  minPerUserTokS: 20,
};

const rtx4090: HardwareSpec = { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 165, reservePct: 5, overheadGB: 1 };

function state(overrides: Partial<CalcState> = {}): CalcState {
  return {
    model: makeSpec(),
    quant: { weight: 'bf16', kv: 'fp16' },
    hardware: rtx4090,
    workload: { contextTokens: 8192, concurrentUsers: 1 },
    runtime: 'generic',
    ...overrides,
  };
}

describe('evaluateHardwareSizingRow', () => {
  it('fails on TP split alone when everything else would pass', () => {
    // 6 attention heads: divisible by 1 and 2, not by 4. Tiny dense model easily fits and is fast.
    const tiny = makeSpec({
      params: 1e9,
      activeParams: 1e9,
      numKvHeads: 2,
      ffn: { intermediateSize: 4096, numAttentionHeads: 6, tieEmbeddings: false },
    });
    const s = state({ model: tiny, hardware: { ...rtx4090, gpuCount: 4 } });
    const result = calculate(s);
    const evaluation = evaluateHardwareSizingRow(result, { ...baseOptions, minPerUserTokS: 0 });
    expect(evaluation.qualifies).toBe(false);
    expect(evaluation.failReason).toBe('TP split invalid');
  });

  it('fails on memory alone: a model far too big for the GPU', () => {
    // Llama-3-70B-scale BF16 weights (~141GB) vs a single 24GB RTX 4090.
    const s = state();
    const result = calculate(s);
    const evaluation = evaluateHardwareSizingRow(result, { ...baseOptions, minPerUserTokS: 0 });
    expect(evaluation.qualifies).toBe(false);
    expect(evaluation.failReason).toMatch(/^short \d+(\.\d+)? GB$/);
  });

  it('fails on tok/s alone when it fits and TP is fine but the target is unreachable', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const s = state({ model: tiny, hardware: { ...rtx4090, gpuCount: 1 } });
    const result = calculate(s);
    expect(result.runs).toBe(true);
    const evaluation = evaluateHardwareSizingRow(result, { ...baseOptions, minPerUserTokS: 1_000_000 });
    expect(evaluation.qualifies).toBe(false);
    expect(evaluation.failReason).toMatch(/tok\/s < 1,000,000/);
  });

  it('fails on TTFT alone when fit and tok/s both pass', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const s = state({ model: tiny, hardware: { ...rtx4090, gpuCount: 1 } });
    const result = calculate(s);
    expect(result.runs).toBe(true);
    const evaluation = evaluateHardwareSizingRow(result, { ...baseOptions, minPerUserTokS: 0, maxTtftSeconds: 1e-9 });
    expect(evaluation.qualifies).toBe(false);
    expect(evaluation.failReason).toMatch(/^TTFT .+ > .+$/);
  });

  it('qualifies when every check passes', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const s = state({ model: tiny, hardware: { ...rtx4090, gpuCount: 1 } });
    const result = calculate(s);
    const evaluation = evaluateHardwareSizingRow(result, { ...baseOptions, minPerUserTokS: 0 });
    expect(evaluation.qualifies).toBe(true);
    expect(evaluation.failReason).toBeUndefined();
    expect(evaluation.gap).toBe(0);
  });
});

describe('rankHardwareSizingRows', () => {
  function row(overrides: Partial<HardwareSizingRow>): HardwareSizingRow {
    const gpu = findGpuPreset('RTX 4090')!;
    const s = state({ hardware: { ...rtx4090, gpuCount: 1 } });
    const result = calculate(s);
    return {
      gpu,
      gpuCount: 1,
      totalVramGB: gpu.vramGB,
      state: s,
      result,
      qualifies: true,
      unusual: false,
      gap: 0,
      ...overrides,
    };
  }

  describe("'smallest' (the default)", () => {
    it('sorts by total VRAM ascending regardless of price', () => {
      const small = row({ totalVramGB: 24 });
      const big = row({ totalVramGB: 96 });
      expect(rankHardwareSizingRows([big, small])).toEqual([small, big]);
      expect(rankHardwareSizingRows([big, small], 'smallest')).toEqual([small, big]);
    });

    it('never buries a small unpriced GPU under a big priced one (the review regression)', () => {
      // A single unpriced-but-owned RTX 4090 vs an 8x H100 cluster with a known cloud rate.
      const single4090 = row({ totalVramGB: 24 });
      const h100Cluster = row({ totalVramGB: 640, cost: { costPerHour: 26, atCurrentUsers: undefined, atMaxUsers: undefined } });
      expect(rankHardwareSizingRows([h100Cluster, single4090], 'smallest')).toEqual([single4090, h100Cluster]);
    });

    it('breaks a VRAM tie by $/hour when both rows have one', () => {
      const cheap = row({ totalVramGB: 80, cost: { costPerHour: 1, atCurrentUsers: undefined, atMaxUsers: undefined } });
      const pricey = row({ totalVramGB: 80, cost: { costPerHour: 5, atCurrentUsers: undefined, atMaxUsers: undefined } });
      expect(rankHardwareSizingRows([pricey, cheap], 'smallest')).toEqual([cheap, pricey]);
    });
  });

  describe("'cheapest'", () => {
    it('sorts priced rows by price ascending', () => {
      const cheap = row({ cost: { costPerHour: 1, atCurrentUsers: undefined, atMaxUsers: undefined } });
      const pricey = row({ cost: { costPerHour: 5, atCurrentUsers: undefined, atMaxUsers: undefined } });
      expect(rankHardwareSizingRows([pricey, cheap], 'cheapest')).toEqual([cheap, pricey]);
    });

    it('sorts unpriced rows by total VRAM ascending', () => {
      const small = row({ totalVramGB: 24 });
      const big = row({ totalVramGB: 96 });
      expect(rankHardwareSizingRows([big, small], 'cheapest')).toEqual([small, big]);
    });

    it('puts every priced row ahead of every unpriced row', () => {
      const unpriced = row({ totalVramGB: 24 });
      const priced = row({ totalVramGB: 999, cost: { costPerHour: 100, atCurrentUsers: undefined, atMaxUsers: undefined } });
      expect(rankHardwareSizingRows([unpriced, priced], 'cheapest')).toEqual([priced, unpriced]);
    });
  });
});

describe('sizeHardware', () => {
  it('every qualifying row actually qualifies and runs', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const { qualifying } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: 4 }, { ...baseOptions, minPerUserTokS: 1 });
    expect(qualifying.length).toBeGreaterThan(0);
    for (const row of qualifying) {
      expect(row.qualifies).toBe(true);
      expect(row.result.runs).toBe(true);
    }
  });

  it("qualifying rows default to 'smallest' (total VRAM ascending) regardless of price", () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const { qualifying } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: 4 }, { ...baseOptions, minPerUserTokS: 1 });
    for (let i = 1; i < qualifying.length; i++) expect(qualifying[i].totalVramGB).toBeGreaterThanOrEqual(qualifying[i - 1].totalVramGB);
  });

  it("options.sort: 'cheapest' ranks priced rows by $/hour ahead of unpriced rows by VRAM", () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const { qualifying } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: 4 }, { ...baseOptions, minPerUserTokS: 1, sort: 'cheapest' });
    const priced = qualifying.filter((r) => r.cost !== undefined);
    const unpriced = qualifying.filter((r) => r.cost === undefined);
    for (let i = 1; i < priced.length; i++) expect(priced[i].cost!.costPerHour).toBeGreaterThanOrEqual(priced[i - 1].cost!.costPerHour);
    for (let i = 1; i < unpriced.length; i++) expect(unpriced[i].totalVramGB).toBeGreaterThanOrEqual(unpriced[i - 1].totalVramGB);
    if (priced.length > 0 && unpriced.length > 0) {
      expect(qualifying.indexOf(priced[priced.length - 1])).toBeLessThan(qualifying.indexOf(unpriced[0]));
    }
  });

  it('reports up to 3 near-misses, each with a reason, when a huge model qualifies nowhere', () => {
    const huge = makeSpec({ params: 5e12, activeParams: 5e12 });
    const { qualifying, nearMisses } = sizeHardware(huge, { contextTokens: 8192, concurrentUsers: 1 }, baseOptions);
    expect(qualifying).toEqual([]);
    expect(nearMisses.length).toBeGreaterThan(0);
    expect(nearMisses.length).toBeLessThanOrEqual(3);
    for (const row of nearMisses) {
      expect(row.qualifies).toBe(false);
      expect(row.failReason).toBeDefined();
    }
  });

  it('never includes the Custom placeholder', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const { qualifying, nearMisses } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: 1 }, baseOptions);
    expect([...qualifying, ...nearMisses].some((r) => r.gpu.name === 'Custom')).toBe(false);
  });

  it('respects the vendor filter', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const { qualifying } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: 1 }, { ...baseOptions, minPerUserTokS: 1, vendor: 'nvidia-datacenter' });
    for (const row of qualifying) expect(row.gpu.vendor).toBe('nvidia-datacenter');
  });

  it("a qualifying row's result equals calculate() run directly against its own state", () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const { qualifying } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: 4 }, { ...baseOptions, minPerUserTokS: 1 });
    expect(qualifying.length).toBeGreaterThan(0);
    for (const row of qualifying) {
      expect(calculate(row.state)).toEqual(row.result);
    }
  });
});

describe('scalingStrip', () => {
  it('returns one entry per SCALING_STRIP_USER_COUNTS, in order', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const strip = scalingStrip(tiny, 8192, { ...baseOptions, minPerUserTokS: 1 });
    expect(strip.map((e) => e.users)).toEqual([...SCALING_STRIP_USER_COUNTS]);
  });

  it('each populated entry is the top-ranked qualifying option for sizeHardware at that user count', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const options = { ...baseOptions, minPerUserTokS: 1 };
    const strip = scalingStrip(tiny, 8192, options);
    for (const entry of strip) {
      const { qualifying } = sizeHardware(tiny, { contextTokens: 8192, concurrentUsers: entry.users }, options);
      if (qualifying.length === 0) {
        expect(entry.row).toBeUndefined();
      } else {
        expect(entry.row?.gpu.name).toBe(qualifying[0].gpu.name);
        expect(entry.row?.gpuCount).toBe(qualifying[0].gpuCount);
      }
    }
  });

  it('256 users needs hardware that actually runs at 256 concurrent users', () => {
    const tiny = makeSpec({ params: 1e9, activeParams: 1e9 });
    const strip = scalingStrip(tiny, 8192, { ...baseOptions, minPerUserTokS: 1 });
    const at256 = strip.find((e) => e.users === 256)?.row;
    expect(at256).toBeDefined();
    expect(at256?.result.throughput.aggregateTokS).toBeGreaterThan(0);
  });
});
