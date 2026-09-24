import { describe, expect, it } from 'vitest';
import { calculateCloudCost, cloudCostFor, costPerHour, effectiveAggregateTokS, usdPerMillionOutputTokens } from './cost';
import { calculate } from './fit';
import { findGpuPreset } from './presets/gpus';
import { findModelPreset } from './presets/models';
import { DISABLED_SPECULATIVE } from './speculative';
import type { CalcState } from './types';

describe('costPerHour', () => {
  it('multiplies price by GPU count', () => {
    expect(costPerHour(3.25, 1)).toBe(3.25);
    expect(costPerHour(3.25, 8)).toBeCloseTo(26);
  });
});

describe('usdPerMillionOutputTokens', () => {
  it('$/hr over tok/s, scaled to a million tokens', () => {
    // $26/hr at 1000 tok/s aggregate: 26 / (1000 * 3600) * 1e6 = $7.222.../1M
    expect(usdPerMillionOutputTokens(26, 1000)).toBeCloseTo(7.2222, 3);
  });

  it('is undefined, not Infinity, when there is no throughput', () => {
    expect(usdPerMillionOutputTokens(26, 0)).toBeUndefined();
    expect(usdPerMillionOutputTokens(26, -1)).toBeUndefined();
  });
});

describe('calculateCloudCost', () => {
  const base = { gpuCount: 1, aggregateTokS: 40, aggregateTokSAtMaxUsers: 400 };

  it('is undefined with no price set (hides the cost card)', () => {
    expect(calculateCloudCost({ ...base, usdPerHour: undefined })).toBeUndefined();
    expect(calculateCloudCost({ ...base, usdPerHour: 0 })).toBeUndefined();
  });

  it('computes cost per hour and $/1M tokens at the current user count and at max users', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25 });
    expect(c).toBeDefined();
    expect(c!.costPerHour).toBe(3.25);
    expect(c!.atCurrentUsers).toBeCloseTo((3.25 / (40 * 3600)) * 1e6, 6);
    expect(c!.atMaxUsers).toBeCloseTo((3.25 / (400 * 3600)) * 1e6, 6);
  });

  it('scales cost per hour with GPU count', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25, gpuCount: 8 });
    expect(c!.costPerHour).toBeCloseTo(26);
  });

  it('atMaxUsers is undefined when there is no max-users throughput', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25, aggregateTokSAtMaxUsers: undefined });
    expect(c!.atMaxUsers).toBeUndefined();
  });
});

describe('cloudCostFor (#20: both figures from the same throughput model as calculate())', () => {
  const llama70b = findModelPreset('meta-llama/Llama-3.3-70B-Instruct')!;
  const llama8b = findModelPreset('meta-llama/Llama-3.1-8B-Instruct')!;
  const rtx4090 = findGpuPreset('RTX 4090')!;
  const hardware = {
    gpuName: rtx4090.name,
    gpuCount: 2,
    vramGB: rtx4090.vramGB,
    bandwidthGBs: rtx4090.bandwidthGBs,
    tflopsBf16: rtx4090.tflopsBf16,
    reservePct: 5,
    overheadGB: 1,
    usdPerHour: 2,
  };
  const reviewState = (overrides: Partial<CalcState> = {}): CalcState => ({
    model: llama70b,
    quant: { weight: 'q4_k_m', kv: 'fp16' },
    hardware: { ...hardware, offload: { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 } },
    workload: { contextTokens: 8192, concurrentUsers: 2 },
    runtime: 'vllm',
    speculative: { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'preset', draftModel: llama8b, draftWeightQuant: 'q4_k_m' },
    ...overrides,
  });

  it('the review repro (70B Q4, 2x4090, vLLM, offload + speculative): max-users cost comes from calculate() at maxUsers', () => {
    const s = reviewState();
    const r = calculate(s);
    expect(r.offload.cpuLayers).toBeGreaterThan(0);
    const c = cloudCostFor(s, r)!;
    expect(c.atCurrentUsers).toBe(usdPerMillionOutputTokens(4, r.speculative.throughput.aggregateTokS));
    const atMax = calculate({ ...s, workload: { ...s.workload, concurrentUsers: r.maxUsersAtContext } });
    expect(c.atMaxUsers).toBe(usdPerMillionOutputTokens(4, atMax.speculative.throughput.aggregateTokS));
  });

  it('when the current users ARE the max users, both figures are identical (the reported $155 vs $21 mismatch)', () => {
    const s0 = reviewState();
    const maxU = calculate(s0).maxUsersAtContext;
    const s = reviewState({ workload: { ...s0.workload, concurrentUsers: maxU } });
    const r = calculate(s);
    expect(r.maxUsersAtContext).toBe(maxU);
    const c = cloudCostFor(s, r)!;
    expect(c.atCurrentUsers).toBeDefined();
    expect(c.atMaxUsers).toBe(c.atCurrentUsers);
    // Same with offload and speculation both off (plain GPU throughput).
    const plain = reviewState({
      model: llama8b,
      hardware,
      speculative: DISABLED_SPECULATIVE,
      workload: { contextTokens: 8192, concurrentUsers: 1 },
    });
    const plainMax = calculate(plain).maxUsersAtContext;
    const atPlainMax = { ...plain, workload: { ...plain.workload, concurrentUsers: plainMax } };
    const pc = cloudCostFor(atPlainMax, calculate(atPlainMax))!;
    expect(pc.atMaxUsers).toBe(pc.atCurrentUsers);
  });

  it('effectiveAggregateTokS: speculative aggregate when on, plain (offload-aware) throughput when off', () => {
    const on = calculate(reviewState());
    expect(effectiveAggregateTokS(on)).toBe(on.speculative.throughput.aggregateTokS);
    const off = calculate(reviewState({ speculative: DISABLED_SPECULATIVE }));
    expect(effectiveAggregateTokS(off)).toBe(off.throughput.aggregateTokS);
  });

  it('is undefined with no price, and atMaxUsers is undefined when maxUsers is 0 or unbounded', () => {
    const s = reviewState({ hardware: { ...hardware, usdPerHour: undefined } });
    expect(cloudCostFor(s, calculate(s))).toBeUndefined();
    const plain = reviewState({ hardware, speculative: DISABLED_SPECULATIVE });
    const r = calculate(plain);
    expect(cloudCostFor(plain, { ...r, maxUsersAtContext: 0 })!.atMaxUsers).toBeUndefined();
    expect(cloudCostFor(plain, { ...r, maxUsersAtContext: Number.POSITIVE_INFINITY })!.atMaxUsers).toBeUndefined();
  });
});
