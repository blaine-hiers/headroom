import { describe, expect, it } from 'vitest';
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
});
