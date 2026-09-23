export const DECODE_EFFICIENCY = 0.7;

export interface ThroughputInput {
  /** Weight bytes read per decode step (active params for MoE). */
  activeWeightBytes: number;
  /** KV bytes per request at the working context. */
  kvBytesPerRequest: number;
  concurrentUsers: number;
  /** Per GPU, GB/s. */
  bandwidthGBs: number;
  gpuCount: number;
  efficiency?: number;
}

export interface ThroughputResult {
  perUserTokS: number;
  aggregateTokS: number;
  efficiency: number;
}

/**
 * Bandwidth-bound decode estimate. Each step streams the active weights once
 * plus every user's KV cache. Multi-GPU assumes tensor parallel, so bandwidth adds.
 */
export function decodeThroughput(input: ThroughputInput): ThroughputResult {
  const efficiency = input.efficiency ?? DECODE_EFFICIENCY;
  const users = Math.max(0, Math.floor(input.concurrentUsers));
  const bytesPerStep = input.activeWeightBytes + users * input.kvBytesPerRequest;
  const bandwidth = input.bandwidthGBs * 1e9 * input.gpuCount * efficiency;
  const stepsPerSec = bytesPerStep > 0 && bandwidth > 0 ? bandwidth / bytesPerStep : 0;
  return { perUserTokS: stepsPerSec, aggregateTokS: stepsPerSec * users, efficiency };
}
