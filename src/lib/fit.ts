import { kvBytesForContext, kvBytesPerToken } from './kvcache';
import { decodeThroughput } from './throughput';
import type { CalcResult, CalcState, HardwareSpec, RuntimeKey } from './types';
import { activeParamsDetailed, weightBytes } from './weights';
import { findGpuPreset } from './presets/gpus';
import { SGLANG_MEM_FRACTION_STATIC, VLLM_GPU_MEMORY_UTILIZATION, VLLM_OVERHEAD_ALLOWANCE_GB, kvContextForRuntime } from './runtime';

export const TABLE_CONTEXTS = [2048, 8192, 32768, 131072] as const;

/**
 * Computes the effective usable VRAM per GPU, accounting for macOS GPU wired-memory limits on Apple.
 * macOS caps GPU-wired memory at: 0.67 × RAM for ≤36 GB, 0.75 × RAM above that.
 * These are the observed defaults reported by the MLX and llama.cpp communities; raise them with
 * `sudo sysctl iogpu.wired_limit_mb=<MB>`. See: https://developer.apple.com/forums/thread/752815
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

/**
 * Usable VRAM under a runtime profile. vLLM and SGLang reserve a fixed fraction of VRAM for
 * their own memory pool instead of the generic reserve %; llama.cpp and MLX use the same
 * accounting as `generic` (MLX already gets Apple's wired-memory limit through
 * `effectiveVramGB`, so there is nothing runtime-specific to add).
 */
export function usableBytesForRuntime(hw: HardwareSpec, runtime: RuntimeKey): number {
  switch (runtime) {
    case 'vllm':
      return hw.gpuCount * effectiveVramGB(hw.gpuName, hw.vramGB, hw.appleWiredLimitGB) * 1e9 * VLLM_GPU_MEMORY_UTILIZATION;
    case 'sglang':
      return hw.gpuCount * effectiveVramGB(hw.gpuName, hw.vramGB, hw.appleWiredLimitGB) * 1e9 * SGLANG_MEM_FRACTION_STATIC;
    default:
      return usableBytes(hw);
  }
}

/** Runtime overhead under a runtime profile: the generic overhead, plus vLLM's CUDA-graph/activation allowance. */
export function overheadBytesForRuntime(hw: HardwareSpec, runtime: RuntimeKey): number {
  const base = overheadBytes(hw);
  return runtime === 'vllm' ? base + VLLM_OVERHEAD_ALLOWANCE_GB * 1e9 * hw.gpuCount : base;
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
  const { model, quant, hardware, workload, runtime } = state;
  const users = Math.max(0, Math.floor(workload.concurrentUsers));
  const ctx = Math.max(0, Math.floor(workload.contextTokens));
  const kvCtx = kvContextForRuntime(ctx, runtime); // == ctx for every profile except vLLM's block rounding

  const perToken = kvBytesPerToken(model, quant.kv);
  const perRequest = kvBytesForContext(model, kvCtx, quant.kv);
  const allUsers = perRequest * users;
  const weights = weightBytes(model.params, quant.weight);
  const overhead = overheadBytesForRuntime(hardware, runtime);
  const usable = usableBytesForRuntime(hardware, runtime);
  const fixed = weights + overhead;
  const total = fixed + allUsers;

  const contexts = new Set<number>(TABLE_CONTEXTS);
  contexts.add(ctx);
  const contextTable = [...contexts]
    .sort((a, b) => a - b)
    .map((c) => {
      const kvReq = kvBytesForContext(model, kvContextForRuntime(c, runtime), quant.kv);
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
