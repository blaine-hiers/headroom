import { describe, expect, it } from 'vitest';
import { KV_QUANTS, WEIGHT_QUANTS, defaultWeightQuantFor } from './quant';

describe('quant tables', () => {
  it('has the spec bits per weight', () => {
    expect(WEIGHT_QUANTS.fp32.bitsPerWeight).toBe(32);
    expect(WEIGHT_QUANTS.bf16.bitsPerWeight).toBe(16);
    expect(WEIGHT_QUANTS.fp16.bitsPerWeight).toBe(16);
    expect(WEIGHT_QUANTS.fp8.bitsPerWeight).toBe(8);
    expect(WEIGHT_QUANTS.q8_0.bitsPerWeight).toBe(8.5);
    expect(WEIGHT_QUANTS.q6_k.bitsPerWeight).toBe(6.56);
    expect(WEIGHT_QUANTS.q5_k_m.bitsPerWeight).toBe(5.69);
    expect(WEIGHT_QUANTS.q4_k_m.bitsPerWeight).toBe(4.85);
    expect(WEIGHT_QUANTS.q4_0.bitsPerWeight).toBe(4.5);
    expect(WEIGHT_QUANTS.awq_gptq_4bit.bitsPerWeight).toBe(4.5);
    expect(WEIGHT_QUANTS.iq4_xs.bitsPerWeight).toBe(4.25);
    expect(WEIGHT_QUANTS.nf4.bitsPerWeight).toBe(4.5);
    expect(WEIGHT_QUANTS.q3_k_m.bitsPerWeight).toBe(3.91);
    expect(WEIGHT_QUANTS.q2_k.bitsPerWeight).toBe(3.35);
  });

  it('has KV bytes per element', () => {
    expect(KV_QUANTS.fp16.bytesPerElement).toBe(2);
    expect(KV_QUANTS.bf16.bytesPerElement).toBe(2);
    expect(KV_QUANTS.fp8.bytesPerElement).toBe(1);
    expect(KV_QUANTS.int8.bytesPerElement).toBe(1);
    expect(KV_QUANTS.int4.bytesPerElement).toBe(0.5);
  });

  it('every entry has a label', () => {
    for (const q of [...Object.values(WEIGHT_QUANTS), ...Object.values(KV_QUANTS)]) {
      expect(q.label.length).toBeGreaterThan(0);
    }
  });

  it('defaults the weight quant to the native dtype', () => {
    expect(defaultWeightQuantFor('bf16')).toBe('bf16');
    expect(defaultWeightQuantFor('fp16')).toBe('fp16');
    expect(defaultWeightQuantFor('fp32')).toBe('fp32');
    expect(defaultWeightQuantFor('fp8')).toBe('fp8');
  });
});
