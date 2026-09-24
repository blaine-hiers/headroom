import { describe, expect, it } from 'vitest';
import { DECODE_EFFICIENCY } from './throughput';
import { DEFAULT_OFFLOAD, offloadDecodeThroughput, planOffload, resolveOffload, splitByLayerFraction } from './offload';
import type { OffloadSpec } from './types';

const disabled: OffloadSpec = { enabled: false, systemRamGB: 64, ramBandwidthGBs: 50 };

describe('resolveOffload', () => {
  it('falls back to the disabled default when undefined (old shared links)', () => {
    expect(resolveOffload(undefined)).toEqual(DEFAULT_OFFLOAD);
    expect(DEFAULT_OFFLOAD.enabled).toBe(false);
  });

  it('passes an explicit spec through unchanged', () => {
    const spec: OffloadSpec = { enabled: true, systemRamGB: 128, ramBandwidthGBs: 90 };
    expect(resolveOffload(spec)).toBe(spec);
  });
});

describe('planOffload', () => {
  it('disabled: every layer stays on the GPU, nothing offloaded', () => {
    const plan = planOffload({ weightBytes: 100e9, numLayers: 80, usableGpuBytes: 1e9, offload: disabled });
    expect(plan.gpuLayers).toBe(80);
    expect(plan.cpuLayers).toBe(0);
    expect(plan.gpuWeightBytes).toBe(100e9);
    expect(plan.cpuWeightBytes).toBe(0);
    expect(plan.fitsInRam).toBe(true);
  });

  it('full fit: enough GPU room for every layer offloads 0 layers', () => {
    const offload: OffloadSpec = { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 };
    // 80 layers, 1 GB/layer; 100 GB of GPU room available comfortably covers all 80.
    const plan = planOffload({ weightBytes: 80e9, numLayers: 80, usableGpuBytes: 100e9, offload });
    expect(plan.gpuLayers).toBe(80);
    expect(plan.cpuLayers).toBe(0);
    expect(plan.gpuWeightBytes).toBe(80e9);
    expect(plan.cpuWeightBytes).toBe(0);
    expect(plan.fitsInRam).toBe(true);
  });

  it('partial offload: some layers spill to RAM and fit', () => {
    const offload: OffloadSpec = { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 };
    // 80 layers, 1 GB/layer; only 30 GB of GPU room -> 30 layers on GPU, 50 in RAM (50 GB, fits in 64 GB).
    const plan = planOffload({ weightBytes: 80e9, numLayers: 80, usableGpuBytes: 30e9, offload });
    expect(plan.gpuLayers).toBe(30);
    expect(plan.cpuLayers).toBe(50);
    expect(plan.gpuWeightBytes).toBeCloseTo(30e9, 0);
    expect(plan.cpuWeightBytes).toBeCloseTo(50e9, 0);
    expect(plan.fitsInRam).toBe(true);
  });

  it('does not fit even with RAM: the CPU layers exceed system RAM', () => {
    const offload: OffloadSpec = { enabled: true, systemRamGB: 32, ramBandwidthGBs: 50 };
    // Same split as above (30 on GPU, 50 GB owed to RAM) but RAM is only 32 GB.
    const plan = planOffload({ weightBytes: 80e9, numLayers: 80, usableGpuBytes: 30e9, offload });
    expect(plan.gpuLayers).toBe(30);
    expect(plan.cpuLayers).toBe(50);
    expect(plan.fitsInRam).toBe(false);
  });

  it('clamps to [0, numLayers]: negative GPU room offloads everything', () => {
    const offload: OffloadSpec = { enabled: true, systemRamGB: 1000, ramBandwidthGBs: 50 };
    const plan = planOffload({ weightBytes: 80e9, numLayers: 80, usableGpuBytes: -5e9, offload });
    expect(plan.gpuLayers).toBe(0);
    expect(plan.cpuLayers).toBe(80);
    expect(plan.gpuWeightBytes).toBe(0);
    expect(plan.cpuWeightBytes).toBe(80e9);
  });

  it('zero layers does not divide by zero', () => {
    const offload: OffloadSpec = { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 };
    const plan = planOffload({ weightBytes: 0, numLayers: 0, usableGpuBytes: 10e9, offload });
    expect(plan.gpuLayers).toBe(0);
    expect(plan.cpuLayers).toBe(0);
    expect(Number.isFinite(plan.bytesPerLayer)).toBe(true);
  });
});

describe('splitByLayerFraction', () => {
  it('splits proportionally to the GPU layer fraction', () => {
    expect(splitByLayerFraction(100, 30, 80)).toEqual({ gpu: 37.5, cpu: 62.5 });
  });

  it('all on GPU or numLayers 0 keeps everything on the GPU side', () => {
    expect(splitByLayerFraction(100, 80, 80)).toEqual({ gpu: 100, cpu: 0 });
    expect(splitByLayerFraction(100, 0, 0)).toEqual({ gpu: 100, cpu: 0 });
  });
});

describe('offloadDecodeThroughput', () => {
  it('matches a plain GPU-only estimate when nothing is offloaded', () => {
    const r = offloadDecodeThroughput({
      gpuActiveWeightBytes: 8.03e9 * 2,
      cpuActiveWeightBytes: 0,
      kvBytesPerRequest: 131072 * 2048,
      concurrentUsers: 1,
      bandwidthGBs: 1008,
      gpuCount: 1,
      ramBandwidthGBs: 50,
      gpuEfficiency: DECODE_EFFICIENCY,
    });
    expect(r.perUserTokS).toBeGreaterThan(40);
    expect(r.perUserTokS).toBeLessThan(45);
  });

  it('adding a CPU-resident share slows decode down', () => {
    const allGpu = offloadDecodeThroughput({
      gpuActiveWeightBytes: 40e9,
      cpuActiveWeightBytes: 0,
      kvBytesPerRequest: 1e6,
      concurrentUsers: 1,
      bandwidthGBs: 1000,
      gpuCount: 1,
      ramBandwidthGBs: 50,
      gpuEfficiency: 0.7,
    });
    const halfOffloaded = offloadDecodeThroughput({
      gpuActiveWeightBytes: 20e9,
      cpuActiveWeightBytes: 20e9,
      kvBytesPerRequest: 1e6,
      concurrentUsers: 1,
      bandwidthGBs: 1000,
      gpuCount: 1,
      ramBandwidthGBs: 50,
      gpuEfficiency: 0.7,
    });
    expect(halfOffloaded.perUserTokS).toBeLessThan(allGpu.perUserTokS);
  });

  it('degenerate inputs give 0, not NaN or Infinity', () => {
    const r = offloadDecodeThroughput({
      gpuActiveWeightBytes: 0,
      cpuActiveWeightBytes: 0,
      kvBytesPerRequest: 0,
      concurrentUsers: 0,
      bandwidthGBs: 1000,
      gpuCount: 1,
      ramBandwidthGBs: 50,
      gpuEfficiency: 0.7,
    });
    expect(r.perUserTokS).toBe(0);
    expect(r.aggregateTokS).toBe(0);
  });
});
