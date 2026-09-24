import { decodeThroughput } from './throughput';

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

export interface CloudCostInput {
  /** Blank/undefined hides the cost card — most GPUs and every custom entry start this way. */
  usdPerHour: number | undefined;
  gpuCount: number;
  /** Per GPU, GB/s (HardwareSpec.bandwidthGBs). */
  bandwidthGBs: number;
  /** Weight bytes read per decode step (CalcResult.activeWeightBytes). */
  activeWeightBytes: number;
  /** KV bytes per request at the chosen context (CalcResult.kvBytesPerRequest). */
  kvBytesPerRequest: number;
  /** Decode efficiency, already scaled for the tensor-parallel penalty (CalcResult.throughput.efficiency). */
  efficiency: number;
  /** Aggregate decode tok/s at the chosen concurrent-user count (CalcResult.throughput.aggregateTokS). */
  aggregateTokS: number;
  /** Most concurrent users this hardware fits at the chosen context (CalcResult.maxUsersAtContext). */
  maxUsersAtContext: number;
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

  let atMaxUsers: number | undefined;
  if (Number.isFinite(input.maxUsersAtContext) && input.maxUsersAtContext > 0) {
    const best = decodeThroughput({
      activeWeightBytes: input.activeWeightBytes,
      kvBytesPerRequest: input.kvBytesPerRequest,
      concurrentUsers: input.maxUsersAtContext,
      bandwidthGBs: input.bandwidthGBs,
      gpuCount: input.gpuCount,
      efficiency: input.efficiency,
    });
    atMaxUsers = usdPerMillionOutputTokens(cph, best.aggregateTokS);
  }

  return { costPerHour: cph, atCurrentUsers, atMaxUsers };
}
