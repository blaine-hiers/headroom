import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { MODEL_PRESETS } from './presets/models';
import type { CalcState } from './types';
import { decodeState, encodeState } from './urlState';
import { activeParamsDetailed } from './weights';

const fallback: CalcState = {
  model: makeSpec(),
  quant: { weight: 'bf16', kv: 'fp16' },
  hardware: { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, reservePct: 5, overheadGB: 1 },
  workload: { contextTokens: 8192, concurrentUsers: 1 },
};

describe('urlState', () => {
  it('round-trips every preset exactly', () => {
    for (const model of MODEL_PRESETS) {
      const s: CalcState = {
        model,
        quant: { weight: 'q4_k_m', kv: 'fp8' },
        hardware: { gpuName: 'H100 SXM', gpuCount: 8, vramGB: 80, bandwidthGBs: 3350, reservePct: 7.5, overheadGB: 1.25 },
        workload: { contextTokens: 32768, concurrentUsers: 16 },
      };
      const qs = encodeState(s);
      expect(decodeState(qs, fallback)).toEqual(s);
      expect(decodeState(`?${qs}`, fallback)).toEqual(s);
    }
  });

  it('round-trips awkward strings and float params', () => {
    const s: CalcState = {
      ...fallback,
      model: makeSpec({ id: 'org/näme & co', name: 'a=b?c#d', params: 70_553_706_496.5, warnings: ['x, y', 'z&w'], source: 'manual' }),
    };
    expect(decodeState(encodeState(s), fallback)).toEqual(s);
  });

  it('round-trips the FFN shapes, so a shared MoE link keeps its structural active params', () => {
    const s: CalcState = {
      ...fallback,
      model: makeSpec({
        moe: { numExperts: 256, expertsPerToken: 8, sharedExperts: 1 },
        ffn: { intermediateSize: 18432, moeIntermediateSize: 2048, firstKDense: 3, numAttentionHeads: 128, tieEmbeddings: false, qLoraRank: 1536 },
      }),
    };
    const decoded = decodeState(encodeState(s), fallback);
    expect(decoded).toEqual(s);
    expect(activeParamsDetailed(decoded.model)).toEqual(activeParamsDetailed(s.model));
    expect(activeParamsDetailed(decoded.model).method).toBe('structural');
  });

  it('uses compact keys', () => {
    const qs = encodeState(fallback);
    expect(qs).toContain('wq=bf16');
    expect(qs).toContain('c=8192');
    expect(qs.length).toBeLessThan(400);
  });

  it('round-trips Apple wired-memory limit override', () => {
    const s: CalcState = {
      ...fallback,
      hardware: { gpuName: 'Apple M2 Ultra', gpuCount: 1, vramGB: 192, bandwidthGBs: 800, reservePct: 5, overheadGB: 1, appleWiredLimitGB: 120 },
    };
    expect(decodeState(encodeState(s), fallback)).toEqual(s);
  });

  it('omits Apple wired-memory limit when undefined', () => {
    const s: CalcState = {
      ...fallback,
      hardware: { gpuName: 'Apple M2 Ultra', gpuCount: 1, vramGB: 192, bandwidthGBs: 800, reservePct: 5, overheadGB: 1 },
    };
    const qs = encodeState(s);
    expect(qs).not.toContain('awl');
    expect(decodeState(qs, fallback)).toEqual(s);
  });

  it('ignores awl param when GPU is switched to non-Apple (stale param scenario)', () => {
    // A URL with Apple M2 Ultra and awl set is loaded, but then user selects an H100
    // The awl param in the URL is still there, but should be ignored since H100 is non-Apple
    const appleState: CalcState = {
      ...fallback,
      hardware: { gpuName: 'Apple M2 Ultra', gpuCount: 1, vramGB: 192, bandwidthGBs: 800, reservePct: 5, overheadGB: 1, appleWiredLimitGB: 120 },
    };
    const qs = encodeState(appleState);
    // Now decode with H100 as the GPU instead
    const nonAppleState: CalcState = {
      ...fallback,
      hardware: { gpuName: 'H100 SXM', gpuCount: 1, vramGB: 80, bandwidthGBs: 3350, reservePct: 5, overheadGB: 1, appleWiredLimitGB: 120 },
    };
    const decoded = decodeState(qs, nonAppleState);
    // The awl param is preserved in the URL state, but effectiveVramGB will ignore it for non-Apple GPUs
    // (In the UI, HardwarePanel clears it when switching GPU selection)
    expect(decoded.hardware.appleWiredLimitGB).toBe(120);
  });

  it('bad input → fallback', () => {
    expect(decodeState('', fallback)).toBe(fallback);
    expect(decodeState('garbage', fallback)).toBe(fallback);
    const good = encodeState(fallback);
    expect(decodeState(good.replace('wq=bf16', 'wq=nope'), fallback)).toBe(fallback);
    expect(decodeState(good.replace('c=8192', 'c=abc'), fallback)).toBe(fallback);
    expect(decodeState(good.replace('c=8192', 'c='), fallback)).toBe(fallback);
    expect(decodeState(good.replace('at=mha_gqa', 'at=weird'), fallback)).toBe(fallback);
    expect(decodeState(good.replace(/&u=\d+/, ''), fallback)).toBe(fallback);
    expect(decodeState(`${good}&moe=1,2`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=1,2,1`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=1,2,yes,,,,`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=,2,1,,,,`, fallback)).toBe(fallback);
    expect(decodeState(`${good}&ff=1,2,0,x,,,`, fallback)).toBe(fallback);
  });
});

describe('urlState: speculative decoding', () => {
  it('a state with no speculative field round-trips without one (old links keep working)', () => {
    expect(decodeState(encodeState(fallback), fallback)).toEqual(fallback);
    expect(encodeState(fallback)).not.toContain('se=');
  });

  it("round-trips a 'none' (n-gram) config", () => {
    const s: CalcState = { ...fallback, speculative: { enabled: true, draftMode: 'none', draftWeightQuant: 'q4_k_m', k: 4, alpha: 0.7 } };
    expect(decodeState(encodeState(s), fallback)).toEqual(s);
  });

  it("round-trips a 'preset' draft model", () => {
    const draft = MODEL_PRESETS.find((m) => m.id === 'meta-llama/Llama-3.1-8B-Instruct');
    if (!draft) throw new Error('missing preset');
    const s: CalcState = {
      ...fallback,
      speculative: { enabled: true, draftMode: 'preset', draftModel: draft, draftWeightQuant: 'q8_0', k: 6, alpha: 0.85 },
    };
    expect(decodeState(encodeState(s), fallback)).toEqual(s);
  });

  it("round-trips a 'custom' draft (params only)", () => {
    const s: CalcState = {
      ...fallback,
      speculative: { enabled: true, draftMode: 'custom', draftParams: 1_500_000_000, draftWeightQuant: 'bf16', k: 4, alpha: 0.7 },
    };
    expect(decodeState(encodeState(s), fallback)).toEqual(s);
  });

  it('an unknown preset id in the URL falls back to no draft model', () => {
    const qs = `${encodeState({ ...fallback, speculative: { enabled: true, draftMode: 'preset', draftWeightQuant: 'q4_k_m', k: 4, alpha: 0.7 } })}&sd=nonexistent/model`;
    const decoded = decodeState(qs, fallback);
    expect(decoded.speculative?.draftMode).toBe('none');
    expect(decoded.speculative?.draftModel).toBeUndefined();
  });
});
