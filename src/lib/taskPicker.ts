// Planner step 1 ("Which model for this task?", issue #24): a pure ranking function over the
// bundled catalog. Framework-free — src/ui/planner/TaskPicker.tsx only renders what this returns.

import { calculate } from './fit';
import { formatBytes, formatNumber, formatTokens } from './format';
import { CUSTOM_GPU_NAME, GPU_PRESETS, findGpuPreset } from './presets/gpus';
import type { GpuPreset } from './presets/gpus';
import type { CatalogEntry, TaskTag } from './presets/catalog';
import type { CalcResult, CalcState, HardwareSpec, KvQuantKey, ModelSpec, WeightQuantKey } from './types';

/**
 * Licenses treated as "permissive" by the license filter. Deliberately a short explicit list
 * (not a heuristic over the license string) — the UI states which licenses this is.
 */
export const PERMISSIVE_LICENSES: readonly string[] = ['apache-2.0', 'mit'];

export interface TaskPickerConstraints {
  task: TaskTag;
  /** A GPU_PRESETS name, or 'any' to search the whole bundled table for each model. */
  gpuName: string | 'any';
  gpuCount: number;
  contextTokens: number;
  concurrentUsers: number;
  weightQuant: WeightQuantKey;
  kvQuant: KvQuantKey;
  licenseFilter: 'any' | 'permissive';
  allowMoe: boolean;
}

export const DEFAULT_TASK_PICKER_CONSTRAINTS: TaskPickerConstraints = {
  task: 'chat',
  gpuName: 'RTX 4090',
  gpuCount: 1,
  contextTokens: 8192,
  concurrentUsers: 1,
  weightQuant: 'q4_k_m',
  kvQuant: 'fp16',
  licenseFilter: 'any',
  allowMoe: true,
};

export interface RankedModelRow {
  entry: CatalogEntry;
  hardware: HardwareSpec;
  result: CalcResult;
  /** One-line explanation built from the row's own numbers, e.g. "fits with 18 GB headroom · 41 tok/s · Apache-2.0 · 128K ctx". */
  reason: string;
}

export interface TaskPickerRankResult {
  rows: RankedModelRow[];
  /** Set (and rows empty) when the task/context/license/MoE filter leaves nothing to rank. */
  hint?: string;
}

/** Top rows shown in the Planner's table. */
export const TASK_PICKER_TOP_N = 8;

/** The heuristic in one sentence, for display next to the table. */
export const TASK_PICKER_RULE =
  'Heuristic, not a benchmark: models that actually run are ranked first, then the larger and more capable model (by params), then higher decode speed, then the newer release.';

function hardwareFromGpuPreset(gpu: GpuPreset, gpuCount: number): HardwareSpec {
  return {
    gpuName: gpu.name,
    gpuCount,
    vramGB: gpu.vramGB,
    bandwidthGBs: gpu.bandwidthGBs,
    tflopsBf16: gpu.tflopsBf16,
    reservePct: 5,
    overheadGB: 1,
    usdPerHour: gpu.usdPerHour,
  };
}

function calcStateFor(model: ModelSpec, hardware: HardwareSpec, constraints: TaskPickerConstraints): CalcState {
  return {
    model,
    quant: { weight: constraints.weightQuant, kv: constraints.kvQuant },
    hardware,
    workload: { contextTokens: constraints.contextTokens, concurrentUsers: constraints.concurrentUsers },
    runtime: 'generic',
  };
}

/**
 * Picks the hardware to run this model's calculation on. A named preset is used as-is; 'any'
 * searches the whole bundled GPU table (smallest VRAM first, same order as the Hardware Finder)
 * for the first one this model actually runs on, so the row can still say "fits on an L4" rather
 * than nothing. If nothing in the table runs it, the largest preset is used so the row reports how
 * far short it is instead of picking an arbitrary small one.
 */
function bestHardwareFor(model: ModelSpec, constraints: TaskPickerConstraints): { hardware: HardwareSpec; result: CalcResult } {
  if (constraints.gpuName !== 'any') {
    const preset = findGpuPreset(constraints.gpuName) ?? GPU_PRESETS.find((g) => g.name !== CUSTOM_GPU_NAME)!;
    const hardware = hardwareFromGpuPreset(preset, constraints.gpuCount);
    return { hardware, result: calculate(calcStateFor(model, hardware, constraints)) };
  }
  const bySize = GPU_PRESETS.filter((g) => g.name !== CUSTOM_GPU_NAME).sort((a, b) => a.vramGB - b.vramGB);
  let fallback: { hardware: HardwareSpec; result: CalcResult } | undefined;
  for (const gpu of bySize) {
    const hardware = hardwareFromGpuPreset(gpu, constraints.gpuCount);
    const result = calculate(calcStateFor(model, hardware, constraints));
    fallback = { hardware, result };
    if (result.runs) return fallback;
  }
  return fallback!; // bySize is never empty
}

function licenseLabel(license: string): string {
  const KNOWN: Record<string, string> = {
    'apache-2.0': 'Apache-2.0',
    mit: 'MIT',
    gemma: 'Gemma',
    llama3: 'Llama 3',
    'llama3.1': 'Llama 3.1',
    'llama3.2': 'Llama 3.2',
    'llama3.3': 'Llama 3.3',
    llama4: 'Llama 4',
    'modified-mit': 'Modified MIT',
    'nvidia-open': 'NVIDIA Open',
  };
  return KNOWN[license] ?? license;
}

function describeRow(entry: CatalogEntry, result: CalcResult): string {
  const fitWord = result.fits ? 'fits' : result.runs ? 'runs with CPU/RAM offload' : 'does not fit';
  const headroomAbs = formatBytes(Math.abs(result.headroomBytes));
  const headroomPhrase = result.headroomBytes >= 0 ? `${headroomAbs} headroom` : `${headroomAbs} short`;
  const tokS = result.throughput.perUserTokS;
  const tokPhrase = `${formatNumber(tokS, tokS < 10 ? 1 : 0)} tok/s`;
  return [fitWord + ' with ' + headroomPhrase, tokPhrase, licenseLabel(entry.meta.license), `${formatTokens(entry.spec.maxPositionEmbeddings)} ctx`].join(' · ');
}

function compareRows(a: RankedModelRow, b: RankedModelRow): number {
  if (a.result.runs !== b.result.runs) return a.result.runs ? -1 : 1;
  if (a.entry.spec.params !== b.entry.spec.params) return b.entry.spec.params - a.entry.spec.params;
  if (a.entry.spec.activeParams !== b.entry.spec.activeParams) return b.entry.spec.activeParams - a.entry.spec.activeParams;
  const tokDiff = b.result.throughput.perUserTokS - a.result.throughput.perUserTokS;
  if (tokDiff !== 0) return tokDiff;
  return b.entry.meta.releaseDate.localeCompare(a.entry.meta.releaseDate);
}

/**
 * Ranks the catalog for one task under a set of hardware/quant/license/MoE constraints.
 * 1. Filter to models tagged with `task` whose `maxPositionEmbeddings` covers the needed context,
 *    matching the license filter and MoE-allowed setting.
 * 2. Run the existing `calculate()` for each on the chosen (or best-fitting "any") hardware.
 * 3. Rank: runs first, then larger total params, then larger active params, then higher decode
 *    tok/s, then newer release date. Ties beyond that keep catalog order (stable sort).
 */
export function rankModelsForTask(catalog: readonly CatalogEntry[], constraints: TaskPickerConstraints): TaskPickerRankResult {
  const candidates = catalog.filter((entry) => {
    if (!entry.meta.tags.includes(constraints.task)) return false;
    if (entry.spec.maxPositionEmbeddings < constraints.contextTokens) return false;
    if (constraints.licenseFilter === 'permissive' && !PERMISSIVE_LICENSES.includes(entry.meta.license)) return false;
    if (!constraints.allowMoe && entry.spec.moe) return false;
    return true;
  });

  if (candidates.length === 0) {
    return {
      rows: [],
      hint: `No bundled model tagged "${constraints.task}" supports ${formatTokens(constraints.contextTokens)} of context under these filters. Try a shorter context, "any license", or allowing MoE models.`,
    };
  }

  const rows = candidates.map((entry) => {
    const { hardware, result } = bestHardwareFor(entry.spec, constraints);
    return { entry, hardware, result, reason: describeRow(entry, result) };
  });
  rows.sort(compareRows);
  return { rows: rows.slice(0, TASK_PICKER_TOP_N) };
}
