import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { calculate, maxContext, maxUsers, overheadBytes, usableBytes } from './fit';
import { findGpuPreset } from './presets/gpus';
import { findModelPreset } from './presets/models';
import type { CalcState, HardwareSpec } from './types';

const h100x4: HardwareSpec = { gpuName: 'H100 SXM', gpuCount: 4, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5, reservePct: 5, overheadGB: 1 };

function state(overrides: Partial<CalcState> = {}): CalcState {
  return {
    model: makeSpec(),
    quant: { weight: 'bf16', kv: 'fp16' },
    hardware: h100x4,
    workload: { contextTokens: 8192, concurrentUsers: 4 },
    ...overrides,
  };
}

describe('usable and overhead', () => {
  it('usable = count × GB × 1e9 × (1 − reserve)', () => {
    expect(usableBytes(h100x4)).toBeCloseTo(4 * 80e9 * 0.95, 0);
    expect(usableBytes({ ...h100x4, reservePct: 0 })).toBe(320e9);
  });
  it('overhead = GB × 1e9 × count', () => {
    expect(overheadBytes(h100x4)).toBe(4e9);
  });
});

describe('maxUsers', () => {
  it('floors the free bytes over KV per request', () => {
    expect(maxUsers(100, 10, 30)).toBe(3);
    expect(maxUsers(100, 10, 90)).toBe(1);
  });
  it('is 0 when fixed memory meets or exceeds usable', () => {
    expect(maxUsers(100, 100, 1)).toBe(0);
    expect(maxUsers(100, 150, 1)).toBe(0);
  });
  it('is unbounded when KV per request is 0', () => {
    expect(maxUsers(100, 10, 0)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe('maxContext', () => {
  it('divides free bytes by users × bytes/token', () => {
    expect(maxContext(1000, 0, 2, 10, 1_000_000)).toBe(50);
    expect(maxContext(1000, 0, 0, 10, 1_000_000)).toBe(100); // users clamps to 1
  });
  it('is capped at maxPositionEmbeddings', () => {
    expect(maxContext(1e15, 0, 1, 1, 131072)).toBe(131072);
    expect(maxContext(1e9, 0, 1, 0, 4096)).toBe(4096);
  });
  it('is 0 when nothing is free', () => {
    expect(maxContext(10, 20, 1, 1, 4096)).toBe(0);
  });
});

describe('calculate', () => {
  it('fills every field for Llama 3 70B on 4× H100', () => {
    const r = calculate(state());
    expect(r.kvBytesPerToken).toBe(327_680);
    expect(r.kvBytesPerRequest).toBe(2_684_354_560);
    expect(r.kvBytesAllUsers).toBe(4 * 2_684_354_560);
    expect(r.weightBytes).toBe(141.2e9);
    expect(r.overheadBytes).toBe(4e9);
    expect(r.usableBytes).toBeCloseTo(304e9, 0);
    expect(r.totalBytes).toBeCloseTo(141.2e9 + 4e9 + 4 * 2_684_354_560, 0);
    expect(r.headroomBytes).toBeCloseTo(r.usableBytes - r.totalBytes, 0);
    expect(r.fits).toBe(true);
    expect(r.maxUsersAtContext).toBe(Math.floor((304e9 - 145.2e9) / 2_684_354_560));
    expect(r.maxContextForUsers).toBe(Math.min(131072, Math.floor((304e9 - 145.2e9) / (4 * 327_680))));
    expect(r.throughput.efficiency).toBe(0.7);
    expect(r.throughput.aggregateTokS).toBeCloseTo(r.throughput.perUserTokS * 4);
    // Prefill/TTFT: makeSpec() has no ffn, so the attention term falls back to hiddenSize.
    const expectedFlops = 2 * 70.6e9 * 8192 + 2 * 80 * 8192 * 8192 * 8192;
    expect(r.prefill.headsSource).toBe('hiddenSize-fallback');
    expect(r.prefill.mfu).toBe(0.4);
    expect(r.prefill.flops).toBe(expectedFlops);
    expect(r.prefill.ttftSeconds).toBeCloseTo(expectedFlops / (989.5e12 * 4 * 0.4), 6);
  });

  it('context table has 2K/8K/32K/128K, adding the chosen context sorted', () => {
    expect(calculate(state()).contextTable.map((r) => r.contextTokens)).toEqual([2048, 8192, 32768, 131072]);
    const t = calculate(state({ workload: { contextTokens: 16384, concurrentUsers: 1 } })).contextTable;
    expect(t.map((r) => r.contextTokens)).toEqual([2048, 8192, 16384, 32768, 131072]);
    expect(t[0].kvBytesPerRequest).toBe(671_088_640);
    expect(t[4].kvBytesPerRequest).toBe(42_949_672_960);
    expect(t[4].maxUsers).toBe(Math.floor((304e9 - 145.2e9) / 42_949_672_960));
  });

  it('does not fit when weights exceed VRAM: maxUsers 0, headroom negative', () => {
    const rtx4090: HardwareSpec = { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 82.6, reservePct: 5, overheadGB: 1 };
    const r = calculate(state({ hardware: rtx4090, workload: { contextTokens: 2048, concurrentUsers: 1 } }));
    expect(r.fits).toBe(false);
    expect(r.headroomBytes).toBeLessThan(0);
    expect(r.maxUsersAtContext).toBe(0);
    expect(r.maxContextForUsers).toBe(0);
    expect(r.contextTable.every((row) => row.maxUsers === 0)).toBe(true);
  });

  it('fits boundary: total exactly equal to usable fits', () => {
    // weights 20e9 − 4096 B, overhead 0, KV 4 B/token × 1024 tokens = 4096 B per user
    const model = makeSpec({ params: (20e9 - 4096) / 2, numLayers: 1, numKvHeads: 1, headDim: 1 });
    const hw: HardwareSpec = { gpuName: 'x', gpuCount: 1, vramGB: 20, bandwidthGBs: 100, tflopsBf16: 100, reservePct: 0, overheadGB: 0 };
    const one = calculate(state({ model, hardware: hw, workload: { contextTokens: 1024, concurrentUsers: 1 } }));
    expect(one.totalBytes).toBe(20e9);
    expect(one.usableBytes).toBe(20e9);
    expect(one.headroomBytes).toBe(0);
    expect(one.fits).toBe(true);
    expect(one.maxUsersAtContext).toBe(1);
    const two = calculate(state({ model, hardware: hw, workload: { contextTokens: 1024, concurrentUsers: 2 } }));
    expect(two.fits).toBe(false);
    expect(two.headroomBytes).toBe(-4096);
  });

  it('caps maxContext at maxPositionEmbeddings when memory is plentiful', () => {
    const b200x8: HardwareSpec = { gpuName: 'B200', gpuCount: 8, vramGB: 192, bandwidthGBs: 8000, tflopsBf16: 1125, reservePct: 5, overheadGB: 1 };
    const model = makeSpec({ params: 8e9, maxPositionEmbeddings: 8192 });
    expect(calculate(state({ model, hardware: b200x8 })).maxContextForUsers).toBe(8192);
  });

  it('throughput sanity: Llama 3.1 8B preset BF16 on a 4090 ≈ 40–45 tok/s', () => {
    const model = findModelPreset('Llama 3.1 8B');
    const gpu = findGpuPreset('RTX 4090');
    if (!model || !gpu) throw new Error('missing preset');
    const r = calculate({
      model,
      quant: { weight: 'bf16', kv: 'fp16' },
      hardware: { gpuName: gpu.name, gpuCount: 1, vramGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs, tflopsBf16: gpu.tflopsBf16, reservePct: 5, overheadGB: 1 },
      workload: { contextTokens: 2048, concurrentUsers: 1 },
    });
    expect(r.throughput.perUserTokS).toBeGreaterThan(40);
    expect(r.throughput.perUserTokS).toBeLessThan(45);
  });

  it('MoE throughput reads only active weights', () => {
    const dense = makeSpec({ params: 30e9 });
    const moe = makeSpec({ params: 30e9, moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 } });
    const a = calculate(state({ model: dense })).throughput.perUserTokS;
    const b = calculate(state({ model: moe })).throughput.perUserTokS;
    expect(b).toBeGreaterThan(a);
    expect(calculate(state({ model: moe })).weightBytes).toBe(60e9); // all experts resident
  });
});
