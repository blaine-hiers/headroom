import { calculate } from './fit';
import { CUSTOM_GPU_NAME, GPU_PRESETS } from './presets/gpus';
import type { CalcResult, CalcState } from './types';
import type { GpuPreset } from './presets/gpus';

/** GPU counts the finder tries for each preset, smallest first. */
export const HARDWARE_FINDER_COUNTS = [1, 2, 4, 8] as const;

/** Consumer GPUs beyond this count are flagged unusual (no vendor-supported multi-GPU interconnect at this scale). */
export const CONSUMER_UNUSUAL_GPU_COUNT = 2;

export interface HardwareFinderRow {
  gpu: GpuPreset;
  gpuCount: number;
  totalVramGB: number;
  result: CalcResult;
  /** A consumer-tier GPU at more than CONSUMER_UNUSUAL_GPU_COUNT is an unusual setup. */
  unusual: boolean;
}

/**
 * Reverse hardware search: for the current model, quant and workload (`state`, minus its own
 * hardware selection), finds the smallest GPU count in HARDWARE_FINDER_COUNTS that fits for
 * each GPU in GPU_PRESETS (excluding the Custom placeholder, which is not a real preset to
 * recommend). Each candidate re-runs the existing `calculate()` with the hardware swapped to
 * that preset/count — keeping the user's reserve %, overhead and runtime — so every other
 * feature's rules (runtime memory accounting, Apple wired-memory limit, tensor-parallel
 * checks, prefill) apply automatically. A count whose tensor-parallel split doesn't divide
 * evenly (`tensorParallel.headsDivisible` / `kvHeadsSplitValid`) does not count as fitting.
 * GPUs where no count in the search set fits are omitted.
 */
export function findFittingHardware(state: CalcState): HardwareFinderRow[] {
  const rows: HardwareFinderRow[] = [];
  for (const gpu of GPU_PRESETS) {
    if (gpu.name === CUSTOM_GPU_NAME) continue;
    for (const gpuCount of HARDWARE_FINDER_COUNTS) {
      const candidate: CalcState = {
        ...state,
        hardware: {
          ...state.hardware,
          gpuName: gpu.name,
          gpuCount,
          vramGB: gpu.vramGB,
          bandwidthGBs: gpu.bandwidthGBs,
          tflopsBf16: gpu.tflopsBf16,
          // A non-Apple GPU has no wired-memory limit; a user's override for a different Apple
          // GPU may not apply to this one either, so only carry it over onto another Apple part.
          appleWiredLimitGB: gpu.vendor === 'apple' ? state.hardware.appleWiredLimitGB : undefined,
        },
      };
      const result = calculate(candidate);
      const tpOk = result.tensorParallel.headsDivisible && result.tensorParallel.kvHeadsSplitValid;
      if (result.fits && tpOk) {
        rows.push({
          gpu,
          gpuCount,
          totalVramGB: gpu.vramGB * gpuCount,
          result,
          unusual: gpu.vendor === 'nvidia-consumer' && gpuCount > CONSUMER_UNUSUAL_GPU_COUNT,
        });
        break;
      }
    }
  }
  return rows;
}
