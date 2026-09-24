// URL persistence for the Planner's TaskPicker (issue #24), separate from the Calculator's own
// urlState.ts. Every key is prefixed `pt_` so it can never collide with the Calculator's keys or
// HardwareSizing's `ph_` keys (issue #25), and an old link (with no `pt_` keys at all) decodes to
// `undefined` — no TaskPicker state — leaving every existing shared link, including `c2`/`c3`
// compare links, unchanged.

import { MAX_CATALOG_CONTEXT, MAX_GPUS, MAX_USERS, MIN_CONTEXT } from './limits';
import { TASK_TAGS } from './presets/catalog';
import { findGpuPreset } from './presets/gpus';
import { KV_QUANTS, WEIGHT_QUANTS } from './quant';
import { DEFAULT_TASK_PICKER_CONSTRAINTS } from './taskPicker';
import type { TaskPickerConstraints } from './taskPicker';
import type { KvQuantKey, WeightQuantKey } from './types';

const PT = {
  task: 'pt_task',
  gpu: 'pt_gpu',
  gpuCount: 'pt_gc',
  context: 'pt_ctx',
  users: 'pt_users',
  weightQuant: 'pt_wq',
  kvQuant: 'pt_kq',
  license: 'pt_lic',
  moe: 'pt_moe',
} as const;

/**
 * Writes (or clears) the TaskPicker's keys on an existing URLSearchParams in place. `undefined`
 * (the user hasn't touched the TaskPicker yet) clears every `pt_` key, so a fresh Planner visit
 * never grows the URL.
 */
export function encodeTaskPickerState(params: URLSearchParams, taskPicker: TaskPickerConstraints | undefined): void {
  if (!taskPicker) {
    for (const key of Object.values(PT)) params.delete(key);
    return;
  }
  params.set(PT.task, taskPicker.task);
  params.set(PT.gpu, taskPicker.gpuName);
  params.set(PT.gpuCount, String(taskPicker.gpuCount));
  params.set(PT.context, String(taskPicker.contextTokens));
  params.set(PT.users, String(taskPicker.concurrentUsers));
  params.set(PT.weightQuant, taskPicker.weightQuant);
  params.set(PT.kvQuant, taskPicker.kvQuant);
  params.set(PT.license, taskPicker.licenseFilter);
  params.set(PT.moe, taskPicker.allowMoe ? '1' : '0');
}

/**
 * Decodes the TaskPicker's constraints from a query string (with or without the leading "?").
 * Missing or invalid fields fall back to `DEFAULT_TASK_PICKER_CONSTRAINTS`'s own value for that
 * field; a missing/invalid `pt_task` (including every link written before this feature existed)
 * returns `undefined` — no TaskPicker state at all, same as a fresh Planner visit.
 */
export function decodeTaskPickerState(search: string): TaskPickerConstraints | undefined {
  const qs = search.startsWith('?') ? search.slice(1) : search;
  const q = new URLSearchParams(qs);
  const task = q.get(PT.task);
  if (task === null || !(TASK_TAGS as readonly string[]).includes(task)) return undefined;

  // Clamped the same way the Calculator's own inputs are (MAX_GPUS, MAX_USERS, MIN_CONTEXT):
  // a crafted `pt_gc`/`pt_ctx`/`pt_users` (huge, negative, or non-finite) must never reach
  // `rankModelsForTask`'s `calculate()` calls unclamped.
  const numClamped = (key: string, fallback: number, lo: number, hi: number): number => {
    const raw = q.get(key);
    if (raw === null) return fallback;
    const n = Number(raw);
    if (!Number.isFinite(n)) return fallback;
    return Math.round(Math.min(hi, Math.max(lo, n)));
  };
  const oneOf = <T extends string>(key: string, allowed: readonly T[], fallback: T): T => {
    const raw = q.get(key);
    return raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;
  };

  const gpuRaw = q.get(PT.gpu);
  const gpuName = gpuRaw !== null && (gpuRaw === 'any' || findGpuPreset(gpuRaw)) ? gpuRaw : DEFAULT_TASK_PICKER_CONSTRAINTS.gpuName;

  return {
    task: task as TaskPickerConstraints['task'],
    gpuName,
    gpuCount: numClamped(PT.gpuCount, DEFAULT_TASK_PICKER_CONSTRAINTS.gpuCount, 1, MAX_GPUS),
    contextTokens: numClamped(PT.context, DEFAULT_TASK_PICKER_CONSTRAINTS.contextTokens, MIN_CONTEXT, MAX_CATALOG_CONTEXT),
    concurrentUsers: numClamped(PT.users, DEFAULT_TASK_PICKER_CONSTRAINTS.concurrentUsers, 1, MAX_USERS),
    weightQuant: oneOf(PT.weightQuant, Object.keys(WEIGHT_QUANTS) as WeightQuantKey[], DEFAULT_TASK_PICKER_CONSTRAINTS.weightQuant),
    kvQuant: oneOf(PT.kvQuant, Object.keys(KV_QUANTS) as KvQuantKey[], DEFAULT_TASK_PICKER_CONSTRAINTS.kvQuant),
    licenseFilter: oneOf(PT.license, ['any', 'permissive'] as const, DEFAULT_TASK_PICKER_CONSTRAINTS.licenseFilter),
    allowMoe: q.get(PT.moe) === null ? DEFAULT_TASK_PICKER_CONSTRAINTS.allowMoe : q.get(PT.moe) === '1',
  };
}
