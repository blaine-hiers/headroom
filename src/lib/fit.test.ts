import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { calculate, effectiveVramGB, maxContext, maxUsers, overheadBytes, usableBytes } from './fit';
import { findGpuPreset } from './presets/gpus';
import { findModelPreset } from './presets/models';
import type { CalcState, HardwareSpec } from './types';

const h100x4: HardwareSpec = { gpuName: 'H100 SXM', gpuCount: 4, vramGB: 80, bandwidthGBs: 3350, reservePct: 5, overheadGB: 1 };

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

describe('effectiveVramGB for Apple GPUs', () => {
  it('non-Apple GPUs return vramGB unchanged', () => {
    expect(effectiveVramGB('H100 SXM', 80)).toBe(80);
    expect(effectiveVramGB('RTX 4090', 24)).toBe(24);
    expect(effectiveVramGB('AMD MI300X', 192)).toBe(192);
  });

  it('Apple GPUs <= 36 GB apply 0.67x wired limit', () => {
    expect(effectiveVramGB('Apple M4 Max', 32, undefined)).toBeCloseTo(32 * 0.67, 1);
    expect(effectiveVramGB('Apple M4 Max', 36, undefined)).toBeCloseTo(36 * 0.67, 1);
  });

  it('Apple GPUs > 36 GB apply 0.75x wired limit', () => {
    expect(effectiveVramGB('Apple M2 Ultra', 192, undefined)).toBeCloseTo(192 * 0.75, 1);
    expect(effectiveVramGB('Apple M3 Ultra', 512, undefined)).toBeCloseTo(512 * 0.75, 1);
  });

  it('Apple GPU with override uses the custom limit', () => {
    expect(effectiveVramGB('Apple M2 Ultra', 192, 120)).toBe(120);
    expect(effectiveVramGB('Apple M4 Max', 32, 28)).toBe(28);
    // Custom limit is clamped to vramGB
    expect(effectiveVramGB('Apple M2 Ultra', 192, 256)).toBe(192);
  });

  it('unknown GPU name (e.g. Custom) returns vramGB unchanged', () => {
    expect(effectiveVramGB('Custom', 24)).toBe(24);
    expect(effectiveVramGB('MyGPU', 100)).toBe(100);
  });
});

describe('usableBytes with Apple wired limit', () => {
  it('applies Apple wired limit for Apple M2 Ultra (> 36 GB)', () => {
    const hw: HardwareSpec = {
      gpuName: 'Apple M2 Ultra',
      gpuCount: 1,
      vramGB: 192,
      bandwidthGBs: 800,
      reservePct: 5,
      overheadGB: 1,
    };
    // 192 * 0.75 * 0.95 * 1e9 = expected usable
    const expected = 192 * 0.75 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });

  it('applies Apple wired limit for Apple M4 Max (<= 36 GB)', () => {
    const hw: HardwareSpec = {
      gpuName: 'Apple M4 Max',
      gpuCount: 1,
      vramGB: 32,
      bandwidthGBs: 546,
      reservePct: 5,
      overheadGB: 1,
    };
    // 32 * 0.67 * 0.95 * 1e9 = expected usable
    const expected = 32 * 0.67 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });

  it('uses custom Apple wired limit when provided', () => {
    const hw: HardwareSpec = {
      gpuName: 'Apple M2 Ultra',
      gpuCount: 1,
      vramGB: 192,
      bandwidthGBs: 800,
      reservePct: 5,
      overheadGB: 1,
      appleWiredLimitGB: 120,
    };
    // 120 * 0.95 * 1e9 = expected usable
    const expected = 120 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });

  it('non-Apple GPUs ignore wired limit override', () => {
    const hw: HardwareSpec = {
      gpuName: 'H100 SXM',
      gpuCount: 1,
      vramGB: 80,
      bandwidthGBs: 3350,
      reservePct: 5,
      overheadGB: 1,
      appleWiredLimitGB: 50,
    };
    // Non-Apple GPU should use full 80 GB, ignoring the override
    const expected = 80 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
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
    const rtx4090: HardwareSpec = { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, reservePct: 5, overheadGB: 1 };
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
    const hw: HardwareSpec = { gpuName: 'x', gpuCount: 1, vramGB: 20, bandwidthGBs: 100, reservePct: 0, overheadGB: 0 };
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
    const b200x8: HardwareSpec = { gpuName: 'B200', gpuCount: 8, vramGB: 192, bandwidthGBs: 8000, reservePct: 5, overheadGB: 1 };
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
      hardware: { gpuName: gpu.name, gpuCount: 1, vramGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs, reservePct: 5, overheadGB: 1 },
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
