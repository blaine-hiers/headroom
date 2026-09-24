import { describe, expect, it } from 'vitest';
import { DEFAULT_MFU, estimateTtft, prefillFlops } from './prefill';

describe('prefillFlops', () => {
  it('golden: matmul + attention term with a known head count', () => {
    // 2 × 1e9 × 1000 = 2e12 matmul; 2 × 10 × 1000² × 10 × 100 = 2e10 attention.
    const r = prefillFlops({ activeParams: 1e9, promptTokens: 1000, numLayers: 10, headDim: 100, numHeads: 10, hiddenSize: 1000 });
    expect(r.flops).toBe(2.02e12);
    expect(r.headsSource).toBe('model');
  });

  it('falls back to hiddenSize when numHeads is absent, and says so', () => {
    // Same shape but numHeads unknown: queryWidth becomes hiddenSize (2000) instead of
    // numHeads × headDim (10 × 100 = 1000), so the attention term doubles.
    const withHeads = prefillFlops({ activeParams: 1e9, promptTokens: 1000, numLayers: 10, headDim: 100, numHeads: 10, hiddenSize: 2000 });
    const withoutHeads = prefillFlops({ activeParams: 1e9, promptTokens: 1000, numLayers: 10, headDim: 100, hiddenSize: 2000 });
    expect(withHeads.headsSource).toBe('model');
    expect(withoutHeads.headsSource).toBe('hiddenSize-fallback');
    expect(withoutHeads.flops).toBe(withHeads.flops + 2e10); // +2 × 10 × 1000² × (2000 − 1000)
  });

  it('treats a numHeads of 0 or negative as absent', () => {
    const r = prefillFlops({ activeParams: 1e9, promptTokens: 1000, numLayers: 10, headDim: 100, numHeads: 0, hiddenSize: 500 });
    expect(r.headsSource).toBe('hiddenSize-fallback');
  });

  it('degenerate inputs give 0, not NaN', () => {
    const r = prefillFlops({ activeParams: 0, promptTokens: 0, numLayers: 0, headDim: 0, hiddenSize: 0 });
    expect(r.flops).toBe(0);
  });
});

describe('estimateTtft', () => {
  it('golden: TTFT = flops / (tflops × 1e12 × gpuCount × MFU)', () => {
    // 1e15 / (100e12 × 2 × 0.4) = 1e15 / 8e13 = 12.5 s.
    const r = estimateTtft({ flops: 1e15, tflopsBf16: 100, gpuCount: 2 });
    expect(r.ttftSeconds).toBe(12.5);
    expect(r.mfu).toBe(DEFAULT_MFU);
  });

  it('honours a custom MFU', () => {
    const r = estimateTtft({ flops: 1e15, tflopsBf16: 100, gpuCount: 2, mfu: 0.5 });
    expect(r.ttftSeconds).toBe(10);
    expect(r.mfu).toBe(0.5);
  });

  it('degenerate inputs give 0, not NaN or Infinity', () => {
    expect(estimateTtft({ flops: 0, tflopsBf16: 100, gpuCount: 1 }).ttftSeconds).toBe(0);
    expect(estimateTtft({ flops: 1e12, tflopsBf16: 0, gpuCount: 1 }).ttftSeconds).toBe(0);
    expect(estimateTtft({ flops: 1e12, tflopsBf16: 100, gpuCount: 0 }).ttftSeconds).toBe(0);
  });
});
