import { describe, expect, it } from 'vitest';
import { DECODE_EFFICIENCY, decodeThroughput } from './throughput';

describe('decodeThroughput', () => {
  it('Llama 3.1 8B BF16 on a 4090 ≈ 40–45 tok/s single user', () => {
    const r = decodeThroughput({
      activeWeightBytes: 8.03e9 * 2,
      kvBytesPerRequest: 131072 * 2048,
      concurrentUsers: 1,
      bandwidthGBs: 1008,
      gpuCount: 1,
    });
    expect(r.perUserTokS).toBeGreaterThan(40);
    expect(r.perUserTokS).toBeLessThan(45);
    expect(r.aggregateTokS).toBe(r.perUserTokS);
    expect(r.efficiency).toBe(DECODE_EFFICIENCY);
  });

  it('bandwidth adds across GPUs; aggregate = per-user × N', () => {
    const one = decodeThroughput({ activeWeightBytes: 1e9, kvBytesPerRequest: 1e8, concurrentUsers: 4, bandwidthGBs: 1000, gpuCount: 1 });
    const two = decodeThroughput({ activeWeightBytes: 1e9, kvBytesPerRequest: 1e8, concurrentUsers: 4, bandwidthGBs: 1000, gpuCount: 2 });
    expect(two.perUserTokS).toBeCloseTo(one.perUserTokS * 2);
    expect(one.perUserTokS).toBeCloseTo((1000e9 * 0.7) / (1e9 + 4e8));
    expect(one.aggregateTokS).toBeCloseTo(one.perUserTokS * 4);
  });

  it('degenerate inputs give 0, not NaN or Infinity', () => {
    const r = decodeThroughput({ activeWeightBytes: 0, kvBytesPerRequest: 0, concurrentUsers: 0, bandwidthGBs: 1000, gpuCount: 1 });
    expect(r.perUserTokS).toBe(0);
    expect(r.aggregateTokS).toBe(0);
  });
});
