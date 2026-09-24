import { describe, expect, it } from 'vitest';
import { plannerReducer } from './plannerState';
import type { PlannerState } from './plannerState';

describe('plannerReducer', () => {
  it('clear then restore brings back the whole state object, not just handoffModelId', () => {
    // Sub-slices added by later steps must survive an undo too.
    const before = { handoffModelId: 'Qwen/Qwen3-32B', taskPicker: { extra: 1 } } as unknown as PlannerState;
    const cleared = plannerReducer(before, { type: 'clear' });
    expect(cleared).toEqual({});
    expect(plannerReducer(cleared, { type: 'restore', state: before })).toEqual(before);
  });
});
