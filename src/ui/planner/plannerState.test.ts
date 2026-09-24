import { describe, expect, it } from 'vitest';
import { MAX_CATALOG_CONTEXT, MAX_USERS, MIN_CONTEXT } from '../../lib';
import { plannerReducer } from './plannerState';
import type { PlannerState } from './plannerState';
import { decodeHardwareSizingState, DEFAULT_HARDWARE_SIZING, encodeHardwareSizingParams } from './plannerState';
import type { HardwareSizingState } from './plannerState';

describe('plannerReducer', () => {
  it('clear then restore brings back the whole state object, not just handoffModelId', () => {
    // Sub-slices added by later steps must survive an undo too.
    const before = { handoffModelId: 'Qwen/Qwen3-32B', taskPicker: { extra: 1 } } as unknown as PlannerState;
    const cleared = plannerReducer(before, { type: 'clear' });
    expect(cleared).toEqual({});
    expect(plannerReducer(cleared, { type: 'restore', state: before })).toEqual(before);
  });

  // Regression for #27 review item 3: step 2 ignored a step-1 handoff whenever it already had an
  // explicit model pick or "use the Calculator's model" on, so "Size hardware" silently did
  // nothing. The handoff must now win by writing straight into hardwareSizing.modelId and
  // clearing useCalculatorModel.
  it('a handoff overrides an existing explicit pick and clears useCalculatorModel (#27 review item 3)', () => {
    const withExplicitPick = plannerReducer({}, { type: 'hardwareSizing/patch', patch: { modelId: 'Qwen/Qwen3-32B', useCalculatorModel: true } });
    const afterHandoff = plannerReducer(withExplicitPick, { type: 'setHandoffModelId', modelId: 'meta-llama/Llama-3.1-8B-Instruct' });
    expect(afterHandoff.handoffModelId).toBe('meta-llama/Llama-3.1-8B-Instruct');
    expect(afterHandoff.hardwareSizing?.modelId).toBe('meta-llama/Llama-3.1-8B-Instruct');
    expect(afterHandoff.hardwareSizing?.useCalculatorModel).toBe(false);
  });

  it('a handoff before step 2 has ever been touched still creates the hardwareSizing slice with the handed-off model', () => {
    const afterHandoff = plannerReducer({}, { type: 'setHandoffModelId', modelId: 'meta-llama/Llama-3.1-8B-Instruct' });
    expect(afterHandoff.hardwareSizing).toEqual({ ...DEFAULT_HARDWARE_SIZING, modelId: 'meta-llama/Llama-3.1-8B-Instruct', useCalculatorModel: false });
  });

  // Regression for #27 review item 4: handoffModelId itself was never persisted to the URL, so
  // a reload/share after "Size hardware" lost the handoff and sized the fallback model instead.
  // Folding the handoff into hardwareSizing.modelId means the existing `phm` key already
  // persists it, with no separate key needed.
  it('a handoff survives encode/decode via the existing phm key (#27 review item 4)', () => {
    const afterHandoff = plannerReducer({}, { type: 'setHandoffModelId', modelId: 'meta-llama/Llama-3.1-8B-Instruct' });
    const params = new URLSearchParams();
    encodeHardwareSizingParams(params, afterHandoff.hardwareSizing);
    const decoded = decodeHardwareSizingState(params.toString());
    expect(decoded?.modelId).toBe('meta-llama/Llama-3.1-8B-Instruct');
    expect(decoded?.useCalculatorModel).toBe(false);
  });
});

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

  // Regression for #27 review item 2: a crafted URL's negative/zero/oversized values used to
  // decode straight through, e.g. ?tab=planner&phu=-3&phc=0&phmt=-5&phtt=-1 rendered -3 users
  // and 0 context instead of clamping like #24's taskPickerUrl.ts does for its own pt_ keys.
  it('clamps a crafted URL instead of passing negative/zero/oversized values through (#27 review item 2)', () => {
    const decoded = decodeHardwareSizingState('tab=planner&phu=-3&phc=0&phmt=-5&phtt=-1');
    expect(decoded?.concurrentUsers).toBe(1);
    expect(decoded?.contextTokens).toBe(MIN_CONTEXT);
    expect(decoded?.minPerUserTokS).toBe(0);
    // maxTtftSeconds <= 0 means "no cap" in the UI, so it must be dropped, not clamped to 0.
    expect(decoded?.maxTtftSeconds).toBeUndefined();
  });

  it('clamps an oversized user count and context to their max bounds', () => {
    const decoded = decodeHardwareSizingState('phu=999999999&phc=999999999');
    expect(decoded?.concurrentUsers).toBe(MAX_USERS);
    expect(decoded?.contextTokens).toBe(MAX_CATALOG_CONTEXT);
  });

  it('drops an unrecognized model id instead of trusting it', () => {
    const decoded = decodeHardwareSizingState('phu=1&phm=not-a-real-model');
    expect(decoded?.modelId).toBeUndefined();
  });

  it('keeps a valid model id', () => {
    const decoded = decodeHardwareSizingState('phu=1&phm=meta-llama%2FLlama-3.1-8B-Instruct');
    expect(decoded?.modelId).toBe('meta-llama/Llama-3.1-8B-Instruct');
  });

  it('a non-finite minPerUserTokS falls back to the default instead of NaN', () => {
    const decoded = decodeHardwareSizingState('phu=1&phmt=not-a-number');
    expect(decoded?.minPerUserTokS).toBe(DEFAULT_HARDWARE_SIZING.minPerUserTokS);
  });
});
