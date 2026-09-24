import { kvBytesForContext, kvBytesPerToken } from './kvcache';
import { decodeThroughput } from './throughput';
import type { CalcResult, CalcState, HardwareSpec } from './types';
import { activeParamsDetailed, weightBytes } from './weights';
import { findGpuPreset } from './presets/gpus';

export const TABLE_CONTEXTS = [2048, 8192, 32768, 131072] as const;

/**
 * Computes the effective usable VRAM per GPU, accounting for macOS GPU wired-memory limits on Apple.
 * macOS caps GPU-wired memory at: 0.67 × RAM for ≤36 GB, 0.75 × RAM above that.
 * See: https://developer.apple.com/forums/thread/752815
 */
export function effectiveVramGB(gpuName: string, vramGB: number, appleWiredLimitGB?: number): number {
  const gpu = findGpuPreset(gpuName);
  if (!gpu || gpu.vendor !== 'apple') {
    return vramGB;
  }

  // Apple GPU: apply wired-memory limit
  // If user provided an override, use it; otherwise use the default OS limit
  if (appleWiredLimitGB !== undefined) {
    return Math.min(vramGB, appleWiredLimitGB);
  }

  // Default macOS wired-memory limit based on total RAM
  // ≤36 GB: 0.67×, >36 GB: 0.75×
  const limit = vramGB <= 36 ? vramGB * 0.67 : vramGB * 0.75;
  return limit;
}

/** usable = gpuCount × effectiveVramGB × 1e9 × (1 − reservePct/100). */
export function usableBytes(hw: HardwareSpec): number {
  const effectiveVram = effectiveVramGB(hw.gpuName, hw.vramGB, hw.appleWiredLimitGB);
  return hw.gpuCount * effectiveVram * 1e9 * (1 - hw.reservePct / 100);
}

/** Runtime overhead across all GPUs: overheadGB × 1e9 × gpuCount. */
export function overheadBytes(hw: HardwareSpec): number {
  return hw.overheadGB * 1e9 * hw.gpuCount;
}

/** floor((usable − fixed) / kvPerRequest); 0 when nothing is free. */
export function maxUsers(usable: number, fixed: number, kvPerRequest: number): number {
  const free = usable - fixed;
  if (!(free > 0)) return 0;
  if (!(kvPerRequest > 0)) return Number.POSITIVE_INFINITY;
  return Math.floor(free / kvPerRequest);
}

/**
 * min(maxPositionEmbeddings, floor((usable − fixed) / (users × bytesPerTokenFull))).
 * Uses the full-attention rate (conservative; sliding layers only lower real use).
 */
export function maxContext(
  usable: number,
  fixed: number,
  users: number,
  bytesPerTokenFull: number,
  maxPositionEmbeddings: number,
): number {
  const free = usable - fixed;
  if (!(free > 0)) return 0;
  if (!(bytesPerTokenFull > 0)) return maxPositionEmbeddings;
  const n = Math.max(1, users);
  return Math.min(maxPositionEmbeddings, Math.floor(free / (n * bytesPerTokenFull)));
}

export function calculate(state: CalcState): CalcResult {
  const { model, quant, hardware, workload } = state;
  const users = Math.max(0, Math.floor(workload.concurrentUsers));
  const ctx = Math.max(0, Math.floor(workload.contextTokens));

  const perToken = kvBytesPerToken(model, quant.kv);
  const perRequest = kvBytesForContext(model, ctx, quant.kv);
  const allUsers = perRequest * users;
  const weights = weightBytes(model.params, quant.weight);
  const overhead = overheadBytes(hardware);
  const usable = usableBytes(hardware);
  const fixed = weights + overhead;
  const total = fixed + allUsers;

  const contexts = new Set<number>(TABLE_CONTEXTS);
  contexts.add(ctx);
  const contextTable = [...contexts]
    .sort((a, b) => a - b)
    .map((c) => {
      const kvReq = kvBytesForContext(model, c, quant.kv);
      return { contextTokens: c, kvBytesPerRequest: kvReq, maxUsers: maxUsers(usable, fixed, kvReq) };
    });

  // Recomputed from the spec (not model.activeParams) so a manual edit stays consistent.
  const active = activeParamsDetailed(model);
  const activeWeightBytes = weightBytes(active.active, quant.weight);
  const throughput = decodeThroughput({
    activeWeightBytes,
    kvBytesPerRequest: perRequest,
    concurrentUsers: users,
    bandwidthGBs: hardware.bandwidthGBs,
    gpuCount: hardware.gpuCount,
  });

  return {
    kvBytesPerToken: perToken,
    kvBytesPerRequest: perRequest,
    kvBytesAllUsers: allUsers,
    weightBytes: weights,
    activeParams: active.active,
    activeParamsMethod: active.method,
    overheadBytes: overhead,
    usableBytes: usable,
    totalBytes: total,
    headroomBytes: usable - total,
    fits: total <= usable,
    maxUsersAtContext: maxUsers(usable, fixed, perRequest),
    maxContextForUsers: maxContext(usable, fixed, users, perToken, model.maxPositionEmbeddings),
    contextTable,
    throughput,
  };
}
