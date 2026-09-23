import { describe, expect, it } from 'vitest';
import { activeParams, weightBytes } from './weights';

describe('weightBytes', () => {
  it('is params × bits / 8', () => {
    expect(weightBytes(8e9, 'bf16')).toBe(16e9);
    expect(weightBytes(8e9, 'fp32')).toBe(32e9);
    expect(weightBytes(8e9, 'fp8')).toBe(8e9);
    expect(weightBytes(8e9, 'q4_k_m')).toBeCloseTo(4.85e9, 0);
    expect(weightBytes(8e9, 'q8_0')).toBe(8.5e9);
  });
});

describe('activeParams', () => {
  it('dense = params', () => {
    expect(activeParams(70e9)).toBe(70e9);
    expect(activeParams(70e9, { numExperts: 1, expertsPerToken: 1, sharedExperts: 0 })).toBe(70e9);
  });

  it('Qwen3-30B-A3B: 128 experts, 8 per token', () => {
    expect(activeParams(30.5e9, { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 })).toBeCloseTo(
      (30.5e9 * 8) / 128,
      0,
    );
  });

  it('DeepSeek-V3: 256 routed + 1 shared, 8 per token', () => {
    expect(activeParams(685e9, { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 })).toBeCloseTo(
      (685e9 * 9) / 257,
      0,
    );
  });

  it('never exceeds params', () => {
    expect(activeParams(10e9, { numExperts: 4, expertsPerToken: 8, sharedExperts: 0 })).toBe(10e9);
  });
});
