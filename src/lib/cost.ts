import { calculate } from './fit';
import type { CalcResult, CalcState } from './types';

/** costPerHour = usdPerHour × gpuCount. */
export function costPerHour(usdPerHour: number, gpuCount: number): number {
  return usdPerHour * gpuCount;
}

/**
 * $ per 1M output tokens = costPerHour / (aggregateTokS × 3600) × 1e6.
 * undefined (never Infinity) when there is no throughput to divide by.
 */
export function usdPerMillionOutputTokens(costPerHourValue: number, aggregateTokS: number): number | undefined {
  if (!(aggregateTokS > 0)) return undefined;
  return (costPerHourValue / (aggregateTokS * 3600)) * 1e6;
}

/**
 * The one aggregate decode rate cost is priced against: the speculative aggregate when
 * speculative decoding is on, otherwise CalcResult.throughput (which is already offload-aware,
 * and 0 when an offloaded split doesn't run).
 */
export function effectiveAggregateTokS(result: Pick<CalcResult, 'throughput' | 'speculative'>): number {
  return result.speculative.enabled ? result.speculative.throughput.aggregateTokS : result.throughput.aggregateTokS;
}

export interface CloudCostInput {
  /** Blank/undefined hides the cost card — most GPUs and every custom entry start this way. */
  usdPerHour: number | undefined;
  gpuCount: number;
  /** effectiveAggregateTokS at the chosen concurrent-user count. */
  aggregateTokS: number;
  /** effectiveAggregateTokS at maxUsersAtContext; undefined when there is no finite, positive max. */
  aggregateTokSAtMaxUsers: number | undefined;
}

export interface CloudCost {
  costPerHour: number;
  /** $ per 1M output tokens at the chosen concurrent-user count. */
  atCurrentUsers: number | undefined;
  /** $ per 1M output tokens at maxUsersAtContext — the best case, since more concurrent users amortize the fixed weight read. */
  atMaxUsers: number | undefined;
}

/** undefined whenever no $/GPU-hour is set — that is what the UI uses to hide the cost card entirely. */
export function calculateCloudCost(input: CloudCostInput): CloudCost | undefined {
  if (input.usdPerHour === undefined || !(input.usdPerHour > 0)) return undefined;
  const cph = costPerHour(input.usdPerHour, input.gpuCount);
  const atCurrentUsers = usdPerMillionOutputTokens(cph, input.aggregateTokS);
  const atMaxUsers = input.aggregateTokSAtMaxUsers === undefined ? undefined : usdPerMillionOutputTokens(cph, input.aggregateTokSAtMaxUsers);
  return { costPerHour: cph, atCurrentUsers, atMaxUsers };
}

/**
 * Cloud cost for a state and its computed result. Both figures come from the same throughput
 * model calculate() uses (offload split, speculative speedup, tensor-parallel penalty): the
 * max-users figure re-runs calculate() with concurrentUsers = maxUsersAtContext and reads its
 * effectiveAggregateTokS, rather than re-deriving a pure-GPU decode estimate.
 */
export function cloudCostFor(state: CalcState, result: CalcResult): CloudCost | undefined {
  const usdPerHour = state.hardware.usdPerHour;
  if (usdPerHour === undefined || !(usdPerHour > 0)) return undefined;
  const maxU = result.maxUsersAtContext;
  const aggregateTokSAtMaxUsers =
    Number.isFinite(maxU) && maxU > 0
      ? effectiveAggregateTokS(calculate({ ...state, workload: { ...state.workload, concurrentUsers: maxU } }))
      : undefined;
  return calculateCloudCost({
    usdPerHour,
    gpuCount: state.hardware.gpuCount,
    aggregateTokS: effectiveAggregateTokS(result),
    aggregateTokSAtMaxUsers,
  });
}
