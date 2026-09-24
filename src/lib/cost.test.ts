import { describe, expect, it } from 'vitest';
import { calculateCloudCost, costPerHour, usdPerMillionOutputTokens } from './cost';

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
  const base = {
    gpuCount: 1,
    bandwidthGBs: 3350,
    activeWeightBytes: 16e9,
    kvBytesPerRequest: 1e8,
    efficiency: 0.7,
    aggregateTokS: 40,
    maxUsersAtContext: 20,
  };

  it('is undefined with no price set (hides the cost card)', () => {
    expect(calculateCloudCost({ ...base, usdPerHour: undefined })).toBeUndefined();
    expect(calculateCloudCost({ ...base, usdPerHour: 0 })).toBeUndefined();
  });

  it('computes cost per hour and $/1M tokens at the current user count', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25 });
    expect(c).toBeDefined();
    expect(c!.costPerHour).toBe(3.25);
    expect(c!.atCurrentUsers).toBeCloseTo((3.25 / (40 * 3600)) * 1e6, 6);
  });

  it('the best case (at max users) costs no more per token than the current-user case', () => {
    // More concurrent users amortize the fixed weight read over more output tokens.
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25 });
    expect(c!.atMaxUsers).toBeDefined();
    expect(c!.atMaxUsers!).toBeLessThanOrEqual(c!.atCurrentUsers!);
  });

  it('scales cost per hour with GPU count', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25, gpuCount: 8 });
    expect(c!.costPerHour).toBeCloseTo(26);
  });

  it('atMaxUsers is undefined when there is no headroom for another user', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25, maxUsersAtContext: 0 });
    expect(c!.atMaxUsers).toBeUndefined();
  });

  it('atMaxUsers is undefined (not NaN) when maxUsersAtContext is unbounded', () => {
    const c = calculateCloudCost({ ...base, usdPerHour: 3.25, maxUsersAtContext: Number.POSITIVE_INFINITY });
    expect(c!.atMaxUsers).toBeUndefined();
  });
});
