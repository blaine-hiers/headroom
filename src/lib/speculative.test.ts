import { describe, expect, it } from 'vitest';
import { findModelPreset } from './presets/models';
import {
  DISABLED_SPECULATIVE,
  expectedTokensPerStep,
  speculativeMemory,
  speculativeThroughput,
} from './speculative';
import type { SpeculativeConfig } from './types';
import { weightBytes } from './weights';

describe('expectedTokensPerStep', () => {
  it('α → 0 gives exactly 1 token per step, for any k', () => {
    expect(expectedTokensPerStep(0, 4)).toBe(1);
    expect(expectedTokensPerStep(0, 0)).toBe(1);
    expect(expectedTokensPerStep(0, 8)).toBe(1);
  });

  it('α → 1 gives exactly k + 1 tokens per step (removable 0/0 limit)', () => {
    expect(expectedTokensPerStep(1, 4)).toBe(5);
    expect(expectedTokensPerStep(1, 0)).toBe(1);
    expect(expectedTokensPerStep(1, 8)).toBe(9);
  });

  it('approaches the α = 1 limit continuously as α → 1', () => {
    expect(expectedTokensPerStep(0.999999, 4)).toBeCloseTo(5, 4);
  });

  it('matches the closed form sum_{i=0}^{k} α^i for a mid-range α', () => {
    const alpha = 0.7;
    const k = 4;
    let sum = 0;
    for (let i = 0; i <= k; i++) sum += alpha ** i;
    expect(expectedTokensPerStep(alpha, k)).toBeCloseTo(sum, 10);
  });
});

describe('speculativeMemory', () => {
  const kv = 'fp16' as const;

  it('disabled: zero everywhere', () => {
    const m = speculativeMemory(DISABLED_SPECULATIVE, 8192, 4, kv);
    expect(m).toEqual({ weightBytes: 0, activeWeightBytes: 0, kvBytesPerRequest: 0, kvBytesAllUsers: 0, kvBytesPerToken: 0, totalBytes: 0 });
  });

  it("draftMode 'none' (n-gram) while enabled: still zero memory", () => {
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'none' };
    const m = speculativeMemory(cfg, 8192, 4, kv);
    expect(m.totalBytes).toBe(0);
  });

  it("draftMode 'preset': weights plus KV cache from the draft's own architecture", () => {
    const draft = findModelPreset('meta-llama/Llama-3.1-8B-Instruct');
    if (!draft) throw new Error('missing preset');
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'preset', draftModel: draft, draftWeightQuant: 'q4_k_m' };
    const m = speculativeMemory(cfg, 8192, 4, kv);
    expect(m.weightBytes).toBeCloseTo(weightBytes(draft.params, 'q4_k_m'), 0);
    expect(m.kvBytesPerRequest).toBeGreaterThan(0);
    expect(m.kvBytesAllUsers).toBeCloseTo(m.kvBytesPerRequest * 4, 0);
    expect(m.kvBytesPerToken).toBeGreaterThan(0);
    expect(m.kvBytesPerRequest).toBeCloseTo(m.kvBytesPerToken * 8192, 0);
    expect(m.totalBytes).toBeCloseTo(m.weightBytes + m.kvBytesAllUsers, 0);
  });

  it("draftMode 'custom': weights only, no KV estimate (architecture unknown)", () => {
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'custom', draftParams: 1e9, draftWeightQuant: 'q4_k_m' };
    const m = speculativeMemory(cfg, 8192, 4, kv);
    expect(m.weightBytes).toBeCloseTo(weightBytes(1e9, 'q4_k_m'), 0);
    expect(m.activeWeightBytes).toBe(m.weightBytes);
    expect(m.kvBytesPerRequest).toBe(0);
    expect(m.kvBytesAllUsers).toBe(0);
    expect(m.totalBytes).toBe(m.weightBytes);
  });

  it('MoE preset draft uses the structural active-params estimate, not the full weight, for activeWeightBytes', () => {
    const draft = findModelPreset('Qwen/Qwen3-30B-A3B');
    if (!draft) throw new Error('missing preset');
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'preset', draftModel: draft, draftWeightQuant: 'bf16' };
    const m = speculativeMemory(cfg, 8192, 1, kv);
    expect(m.activeWeightBytes).toBeLessThan(m.weightBytes);
  });
});

describe('speculativeThroughput', () => {
  const baseline = { perUserTokS: 40, efficiency: 0.7 };
  const hardware = { bandwidthGBs: 1000, gpuCount: 1 };

  it('disabled: passes the baseline straight through, multiplier 1', () => {
    const t = speculativeThroughput(DISABLED_SPECULATIVE, baseline, { activeWeightBytes: 0, kvBytesPerRequest: 0 }, 4, hardware);
    expect(t.perUserTokS).toBe(baseline.perUserTokS);
    expect(t.aggregateTokS).toBe(baseline.perUserTokS * 4);
    expect(t.multiplier).toBe(1);
    expect(t.expectedTokensPerStep).toBe(1);
  });

  it("draftMode 'none' (n-gram): zero draft step time, so speedup is exactly expectedTokensPerStep", () => {
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'none', k: 4, alpha: 0.7 };
    const t = speculativeThroughput(cfg, baseline, { activeWeightBytes: 0, kvBytesPerRequest: 0 }, 1, hardware);
    expect(t.draftStepSeconds).toBe(0);
    expect(t.verifyStepSeconds).toBeCloseTo(t.targetStepSeconds, 12);
    expect(t.multiplier).toBeCloseTo(t.expectedTokensPerStep, 6);
  });

  it('a real draft model step time raises verifyStepSeconds and lowers the multiplier vs n-gram', () => {
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'preset', k: 4, alpha: 0.7 };
    const withDraft = speculativeThroughput(cfg, baseline, { activeWeightBytes: 1e9, kvBytesPerRequest: 1e6 }, 1, hardware);
    const ngram = speculativeThroughput({ ...cfg, draftMode: 'none' }, baseline, { activeWeightBytes: 0, kvBytesPerRequest: 0 }, 1, hardware);
    expect(withDraft.verifyStepSeconds).toBeGreaterThan(ngram.verifyStepSeconds);
    expect(withDraft.multiplier).toBeLessThan(ngram.multiplier);
  });

  it('k = 0 degenerates to the target-only step time regardless of draft cost', () => {
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'preset', k: 0, alpha: 0.7 };
    const t = speculativeThroughput(cfg, baseline, { activeWeightBytes: 1e12, kvBytesPerRequest: 1e12 }, 1, hardware);
    expect(t.verifyStepSeconds).toBeCloseTo(t.targetStepSeconds, 10);
    expect(t.expectedTokensPerStep).toBe(1);
  });

  it('aggregate = perUserTokS × users', () => {
    const cfg: SpeculativeConfig = { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'none', k: 4, alpha: 0.7 };
    const t = speculativeThroughput(cfg, baseline, { activeWeightBytes: 0, kvBytesPerRequest: 0 }, 6, hardware);
    expect(t.aggregateTokS).toBeCloseTo(t.perUserTokS * 6, 6);
  });
});
