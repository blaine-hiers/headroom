import { describe, expect, it } from 'vitest';
import { decodeHardwareSizingState, DEFAULT_HARDWARE_SIZING, encodeHardwareSizingParams } from './plannerState';
import type { HardwareSizingState } from './plannerState';

describe('hardwareSizing URL persistence (#25)', () => {
  it('decodes to undefined when the step has never been touched (no ph keys at all)', () => {
    expect(decodeHardwareSizingState('')).toBeUndefined();
    expect(decodeHardwareSizingState('?tab=planner')).toBeUndefined();
  });

  it('round-trips every field, including sort, through encode/decode', () => {
    const state: HardwareSizingState = {
      modelId: 'meta-llama/Llama-3.1-8B-Instruct',
      useCalculatorModel: true,
      concurrentUsers: 64,
      contextTokens: 16384,
      weightQuant: 'q8_0',
      kvQuant: 'fp8',
      runtime: 'vllm',
      minPerUserTokS: 15,
      maxTtftSeconds: 3.5,
      vendor: 'nvidia-datacenter',
      offloadEnabled: true,
      sort: 'cheapest',
    };
    const params = new URLSearchParams();
    encodeHardwareSizingParams(params, state);
    const decoded = decodeHardwareSizingState(params.toString());
    expect(decoded).toEqual(state);
  });

  it('a link written before the sort control existed decodes to the default sort', () => {
    const params = new URLSearchParams();
    encodeHardwareSizingParams(params, DEFAULT_HARDWARE_SIZING);
    params.delete('phs'); // simulate an older link
    const decoded = decodeHardwareSizingState(params.toString());
    expect(decoded?.sort).toBe('smallest');
  });

  it('an invalid sort value falls back to the default rather than invalidating the whole state', () => {
    const params = new URLSearchParams();
    encodeHardwareSizingParams(params, DEFAULT_HARDWARE_SIZING);
    params.set('phs', 'bogus');
    const decoded = decodeHardwareSizingState(params.toString());
    expect(decoded?.sort).toBe('smallest');
  });

  it('nothing is written for an untouched state (undefined is a no-op)', () => {
    const params = new URLSearchParams([['tab', 'planner']]);
    encodeHardwareSizingParams(params, undefined);
    expect(params.toString()).toBe('tab=planner');
  });
});
