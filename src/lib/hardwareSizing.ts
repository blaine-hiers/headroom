import { calculate } from './fit';
import { cloudCostFor } from './cost';
import type { CloudCost } from './cost';
import { formatNumber, formatSeconds } from './format';
import { CONSUMER_UNUSUAL_GPU_COUNT, HARDWARE_FINDER_COUNTS } from './hardwareFinder';
import { CUSTOM_GPU_NAME, GPU_PRESETS } from './presets/gpus';
import type { GpuPreset, GpuVendor } from './presets/gpus';
import type { CalcResult, CalcState, HardwareSpec, ModelSpec, OffloadSpec, Quant, RuntimeKey, Workload } from './types';

/** One user count and context length to size hardware for. */
export type HardwareSizingLoad = Workload;

/** Everything HardwareSizing's search needs beyond the model and load. */
export interface HardwareSizingOptions {
  quant: Quant;
  runtime: RuntimeKey;
  /** Minimum acceptable per-user decode tok/s. */
  minPerUserTokS: number;
  /** Maximum acceptable time-to-first-token, seconds. Omitted = not checked. */
  maxTtftSeconds?: number;
  /** Restrict the search to one vendor group (same groups as the GPU select); omitted = all vendors. */
  vendor?: GpuVendor;
  /** CPU/RAM layer offload for the candidates; omitted/disabled = off, matching the Calculator's default. */
  offload?: OffloadSpec;
  /** How to order qualifying rows; omitted = 'smallest' (see rankHardwareSizingRows). */
  sort?: HardwareSizingSort;
}

/**
 * 'smallest' (the default): total VRAM ascending, ties broken by $/hour where both rows have
 * one. This is the personal-tool-friendly default — it never buries a qualifying, possibly
 * already-owned single GPU under a priced multi-GPU cluster just because the cluster has a
 * known cloud rate and the single GPU doesn't.
 * 'cheapest': priced rows by $/hour ascending, then unpriced rows by total VRAM ascending —
 * useful once you're actually comparing cloud rental cost.
 */
export type HardwareSizingSort = 'smallest' | 'cheapest';

export interface HardwareSizingRow {
  gpu: GpuPreset;
  gpuCount: number;
  totalVramGB: number;
  /** The exact CalcState calculate() was run against — feeds the "Use" action so it reproduces this row exactly. */
  state: CalcState;
  result: CalcResult;
  qualifies: boolean;
  /** Set when qualifies is false: a short label for why, e.g. "short 6 GB", "14 tok/s < 20", "TP split invalid". */
  failReason?: string;
  cost?: CloudCost;
  /** A consumer-tier GPU at more than CONSUMER_UNUSUAL_GPU_COUNT is an unusual setup. */
  unusual: boolean;
  /**
   * How close a non-qualifying row came to qualifying, as a non-negative fraction (0 once it
   * qualifies). Only meaningful for ordering near-misses against each other, not for comparing
   * across different failure kinds in an absolute sense.
   */
  gap: number;
}

export interface SizeHardwareResult {
  /** Rows that meet every requirement, cheapest-or-smallest first (see rankHardwareSizingRows). */
  qualifying: HardwareSizingRow[];
  /** Up to 3 non-qualifying rows closest to qualifying, for "why did this almost work" context. */
  nearMisses: HardwareSizingRow[];
}

const MAX_NEAR_MISSES = 3;

function buildHardwareSizingState(model: ModelSpec, load: HardwareSizingLoad, gpu: GpuPreset, gpuCount: number, options: HardwareSizingOptions): CalcState {
  const hardware: HardwareSpec = {
    gpuName: gpu.name,
    gpuCount,
    vramGB: gpu.vramGB,
    bandwidthGBs: gpu.bandwidthGBs,
    tflopsBf16: gpu.tflopsBf16,
    reservePct: 5,
    overheadGB: 1,
  };
  if (gpu.usdPerHour !== undefined) hardware.usdPerHour = gpu.usdPerHour;
  if (options.offload) hardware.offload = options.offload;
  return {
    model,
    quant: options.quant,
    hardware,
    workload: { contextTokens: load.contextTokens, concurrentUsers: load.concurrentUsers },
    runtime: options.runtime,
  };
}

/**
 * Qualification for one candidate: runs at the load's user count, tensor-parallel split is
 * valid, per-user tok/s meets the target, and TTFT (if a target is set) does not exceed it.
 * Checked in that order — the first failing check is the one reported, so a row that is both
 * short on memory and slow only ever reports "short N GB".
 */
export function evaluateHardwareSizingRow(result: CalcResult, options: HardwareSizingOptions): { qualifies: boolean; failReason?: string; gap: number } {
  const tpOk = result.tensorParallel.headsDivisible && result.tensorParallel.kvHeadsSplitValid;
  if (!tpOk) return { qualifies: false, failReason: 'TP split invalid', gap: 1 };

  if (!result.runs) {
    const shortBytes = Math.max(0, -result.runHeadroomBytes);
    const shortGB = shortBytes / 1e9;
    const gap = result.usableBytes > 0 ? Math.min(1, shortBytes / result.usableBytes) : 1;
    return { qualifies: false, failReason: `short ${formatNumber(shortGB, shortGB < 10 ? 1 : 0)} GB`, gap };
  }

  const { perUserTokS } = result.throughput;
  if (perUserTokS < options.minPerUserTokS) {
    const gap = options.minPerUserTokS > 0 ? (options.minPerUserTokS - perUserTokS) / options.minPerUserTokS : 1;
    return {
      qualifies: false,
      failReason: `${formatNumber(perUserTokS, perUserTokS < 10 ? 1 : 0)} tok/s < ${formatNumber(options.minPerUserTokS)}`,
      gap,
    };
  }

  if (options.maxTtftSeconds !== undefined && result.prefill.ttftSeconds > options.maxTtftSeconds) {
    const gap = options.maxTtftSeconds > 0 ? (result.prefill.ttftSeconds - options.maxTtftSeconds) / options.maxTtftSeconds : 1;
    return {
      qualifies: false,
      failReason: `TTFT ${formatSeconds(result.prefill.ttftSeconds)} > ${formatSeconds(options.maxTtftSeconds)}`,
      gap,
    };
  }

  return { qualifies: true, gap: 0 };
}

/** See HardwareSizingSort for what each mode means. */
export function rankHardwareSizingRows(rows: readonly HardwareSizingRow[], sort: HardwareSizingSort = 'smallest'): HardwareSizingRow[] {
  const sorted = [...rows];
  if (sort === 'cheapest') {
    sorted.sort((a, b) => {
      const aPriced = a.cost !== undefined;
      const bPriced = b.cost !== undefined;
      if (aPriced && bPriced) return a.cost!.costPerHour - b.cost!.costPerHour;
      if (aPriced !== bPriced) return aPriced ? -1 : 1;
      return a.totalVramGB - b.totalVramGB;
    });
  } else {
    sorted.sort((a, b) => {
      if (a.totalVramGB !== b.totalVramGB) return a.totalVramGB - b.totalVramGB;
      if (a.cost !== undefined && b.cost !== undefined) return a.cost.costPerHour - b.cost.costPerHour;
      return 0; // stable: keeps GPU_PRESETS order for a tie neither rule breaks
    });
  }
  return sorted;
}

/**
 * Sizes hardware for one model at a target load. Reuses the hardware finder's search (every
 * GPU preset x {1, 2, 4, 8}, see hardwareFinder.ts) but qualifies each candidate against the
 * load's requirements instead of just "fits" (see evaluateHardwareSizingRow). For each GPU, the
 * smallest qualifying count wins and the rest of that GPU's counts are skipped; a GPU with no
 * qualifying count in the search set contributes its closest near-miss instead. Vendor-filtered
 * and Custom-excluded exactly like findFittingHardware.
 */
export function sizeHardware(model: ModelSpec, load: HardwareSizingLoad, options: HardwareSizingOptions): SizeHardwareResult {
  const rows: HardwareSizingRow[] = [];
  for (const gpu of GPU_PRESETS) {
    if (gpu.name === CUSTOM_GPU_NAME) continue;
    if (options.vendor !== undefined && gpu.vendor !== options.vendor) continue;

    let bestMiss: HardwareSizingRow | undefined;
    let qualified = false;
    for (const gpuCount of HARDWARE_FINDER_COUNTS) {
      const state = buildHardwareSizingState(model, load, gpu, gpuCount, options);
      const result = calculate(state);
      const evaluation = evaluateHardwareSizingRow(result, options);
      const row: HardwareSizingRow = {
        gpu,
        gpuCount,
        totalVramGB: gpu.vramGB * gpuCount,
        state,
        result,
        qualifies: evaluation.qualifies,
        failReason: evaluation.failReason,
        cost: cloudCostFor(state, result),
        unusual: gpu.vendor === 'nvidia-consumer' && gpuCount > CONSUMER_UNUSUAL_GPU_COUNT,
        gap: evaluation.gap,
      };
      if (evaluation.qualifies) {
        rows.push(row);
        qualified = true;
        break;
      }
      if (!bestMiss || row.gap < bestMiss.gap) bestMiss = row;
    }
    if (!qualified && bestMiss) rows.push(bestMiss);
  }

  const qualifying = rankHardwareSizingRows(rows.filter((r) => r.qualifies), options.sort);
  const nearMisses = rows
    .filter((r) => !r.qualifies)
    .sort((a, b) => a.gap - b.gap)
    .slice(0, MAX_NEAR_MISSES);
  return { qualifying, nearMisses };
}

/** User counts the "how does this scale?" strip shows. */
export const SCALING_STRIP_USER_COUNTS = [1, 4, 16, 64, 256] as const;

export interface ScalingStripEntry {
  users: number;
  /** The cheapest-or-smallest qualifying option at this user count; undefined when nothing qualifies. */
  row: HardwareSizingRow | undefined;
}

/** For the chosen model, the smallest qualifying option at 1, 4, 16, 64 and 256 users, same context per user. */
export function scalingStrip(model: ModelSpec, contextTokens: number, options: HardwareSizingOptions): ScalingStripEntry[] {
  return SCALING_STRIP_USER_COUNTS.map((users) => {
    const { qualifying } = sizeHardware(model, { contextTokens, concurrentUsers: users }, options);
    return { users, row: qualifying[0] };
  });
}
