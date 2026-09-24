import { describe, expect, it } from 'vitest';
import { MAX_CATALOG_CONTEXT, MAX_GPUS, MAX_USERS, MIN_CONTEXT } from './limits';
import { DEFAULT_TASK_PICKER_CONSTRAINTS } from './taskPicker';
import { decodeTaskPickerState, encodeTaskPickerState } from './taskPickerUrl';

describe('TaskPicker URL codec', () => {
  it('round-trips a full set of constraints', () => {
    const constraints = { ...DEFAULT_TASK_PICKER_CONSTRAINTS, task: 'coding' as const, gpuName: 'H100 SXM', gpuCount: 2, allowMoe: false, licenseFilter: 'permissive' as const };
    const params = new URLSearchParams();
    encodeTaskPickerState(params, constraints);
    expect(decodeTaskPickerState(params.toString())).toEqual(constraints);
  });

  it('decodes to undefined when pt_task is absent (an old shared link)', () => {
    expect(decodeTaskPickerState('id=meta-llama%2FLlama-3.3-70B-Instruct&n=Llama')).toBeUndefined();
    expect(decodeTaskPickerState('')).toBeUndefined();
  });

  it('decodes to undefined when pt_task is an unrecognized value', () => {
    expect(decodeTaskPickerState('pt_task=not-a-real-tag')).toBeUndefined();
  });

  it('clears every pt_ key when encoding undefined', () => {
    const params = new URLSearchParams('pt_task=chat&pt_gpu=RTX+4090&tab=planner');
    encodeTaskPickerState(params, undefined);
    expect(params.get('pt_task')).toBeNull();
    expect(params.get('pt_gpu')).toBeNull();
    expect(params.get('tab')).toBe('planner'); // untouched
  });

  it('falls back to the default for a missing individual field', () => {
    const decoded = decodeTaskPickerState('pt_task=chat');
    expect(decoded).toEqual(DEFAULT_TASK_PICKER_CONSTRAINTS);
  });

  it('does not disturb an old c2/c3 compare link with no pt_ keys', () => {
    const qs = 'id=a&n=A&c2=id%3Db%26n%3DB';
    expect(decodeTaskPickerState(qs)).toBeUndefined();
  });

  it('clamps a crafted gpu count, context and user count instead of passing them through', () => {
    const decoded = decodeTaskPickerState('pt_task=chat&pt_gc=999999999&pt_ctx=-500&pt_users=-3');
    expect(decoded?.gpuCount).toBe(MAX_GPUS);
    expect(decoded?.contextTokens).toBe(MIN_CONTEXT);
    expect(decoded?.concurrentUsers).toBe(1);
  });

  it('clamps an oversized user count to MAX_USERS instead of passing it through', () => {
    const decoded = decodeTaskPickerState('pt_task=chat&pt_users=999999999');
    expect(decoded?.concurrentUsers).toBe(MAX_USERS);
  });

  it('clamps an oversized context to the largest context any bundled model supports', () => {
    const decoded = decodeTaskPickerState('pt_task=chat&pt_ctx=999999999999');
    expect(decoded?.contextTokens).toBe(MAX_CATALOG_CONTEXT);
  });

  it('falls back to the default gpu count/context/users for a non-numeric value', () => {
    const decoded = decodeTaskPickerState('pt_task=chat&pt_gc=not-a-number&pt_ctx=nope&pt_users=nope');
    expect(decoded?.gpuCount).toBe(DEFAULT_TASK_PICKER_CONSTRAINTS.gpuCount);
    expect(decoded?.contextTokens).toBe(DEFAULT_TASK_PICKER_CONSTRAINTS.contextTokens);
    expect(decoded?.concurrentUsers).toBe(DEFAULT_TASK_PICKER_CONSTRAINTS.concurrentUsers);
  });

  it('rejects an unknown pt_gpu value, falling back to the default GPU', () => {
    const decoded = decodeTaskPickerState('pt_task=chat&pt_gpu=NotAGpu');
    expect(decoded?.gpuName).toBe(DEFAULT_TASK_PICKER_CONSTRAINTS.gpuName);
  });

  it('accepts "any" and a real GPU preset name for pt_gpu', () => {
    expect(decodeTaskPickerState('pt_task=chat&pt_gpu=any')?.gpuName).toBe('any');
    expect(decodeTaskPickerState('pt_task=chat&pt_gpu=H100+SXM')?.gpuName).toBe('H100 SXM');
  });

  it('rejects an unknown pt_wq/pt_kq value, falling back to the defaults', () => {
    const decoded = decodeTaskPickerState('pt_task=chat&pt_wq=not-a-quant&pt_kq=not-a-quant');
    expect(decoded?.weightQuant).toBe(DEFAULT_TASK_PICKER_CONSTRAINTS.weightQuant);
    expect(decoded?.kvQuant).toBe(DEFAULT_TASK_PICKER_CONSTRAINTS.kvQuant);
  });
});
