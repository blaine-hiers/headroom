import { resolveWeights } from './fileWeights';
import { kvBytesForContext, kvBytesPerToken } from './kvcache';
import { estimateTtft, prefillFlops } from './prefill';
import { offloadDecodeThroughput, planOffload, resolveOffload, splitByLayerFraction } from './offload';
import { checkTensorParallelSplit, tensorParallelEfficiency } from './tensorParallel';
import { DECODE_EFFICIENCY, decodeThroughput } from './throughput';
import type { CalcResult, CalcState, HardwareSpec, RuntimeKey } from './types';
import { activeParamsDetailed } from './weights';
import { findGpuPreset } from './presets/gpus';
import { SGLANG_MEM_FRACTION_STATIC, VLLM_GPU_MEMORY_UTILIZATION, VLLM_KV_BLOCK_TOKENS, VLLM_OVERHEAD_ALLOWANCE_GB, kvContextForRuntime } from './runtime';

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
 *
 * For vLLM, a request's KV reservation rounds UP to the next 16-token block
 * (`kvContextForRuntime`), so solving from the unrounded per-token rate can report a context
 * whose actual (rounded) reservation overshoots `usable`. Rounding the memory-bound answer
 * DOWN to a block multiple guarantees the reported context's real reservation still fits —
 * see the proof in fit.test.ts's "vLLM max-context block rounding" tests. The
 * `maxPositionEmbeddings` cap is never rounded: if the model's own max already fits, there is
 * nothing to shrink.
 */
export function maxContext(
  usable: number,
  fixed: number,
  users: number,
  bytesPerTokenFull: number,
  maxPositionEmbeddings: number,
  runtime: RuntimeKey = 'generic',
): number {
  const free = usable - fixed;
  if (!(free > 0)) return 0;
  if (!(bytesPerTokenFull > 0)) return maxPositionEmbeddings;
  const n = Math.max(1, users);
  const memoryBound = free / (n * bytesPerTokenFull);
  const rounded = runtime === 'vllm' ? Math.floor(memoryBound / VLLM_KV_BLOCK_TOKENS) * VLLM_KV_BLOCK_TOKENS : Math.floor(memoryBound);
  return Math.min(maxPositionEmbeddings, rounded);
}

export function calculate(state: CalcState): CalcResult {
  const { model, quant, hardware, workload, runtime = 'generic' } = state;
  const users = Math.max(0, Math.floor(workload.concurrentUsers));
  const ctx = Math.max(0, Math.floor(workload.contextTokens));
  const kvCtx = kvContextForRuntime(ctx, runtime); // == ctx for every profile except vLLM's block rounding

  // Tensor-parallel split: warns when numAttentionHeads can't divide evenly across gpuCount,
  // and scales KV bytes up when numKvHeads < gpuCount forces KV-head replication.
  const tensorParallel = checkTensorParallelSplit(model, hardware.gpuCount);
  const kvReplication = tensorParallel.kvReplicationFactor;
  const perToken = kvBytesPerToken(model, quant.kv) * kvReplication;
  const perRequest = kvBytesForContext(model, kvCtx, quant.kv) * kvReplication;
  const allUsers = perRequest * users;
  // Recomputed from the spec (not model.activeParams) so a manual edit stays consistent.
  const active = activeParamsDetailed(model);
  const resolved = resolveWeights(model, quant.weight, active.active);
  const weights = resolved.bytes;
  const overhead = overheadBytesForRuntime(hardware, runtime);
  const usable = usableBytesForRuntime(hardware, runtime);
  const fixed = weights + overhead;
  const total = fixed + allUsers;

  // CPU/RAM layer offload (llama.cpp's -ngl): off unless hardware.offload.enabled (see offload.ts).
  // KV stays on the GPU (llama.cpp's default), so only weight bytes are split here.
  const offloadSpec = resolveOffload(hardware.offload);
  const offload = planOffload({
    weightBytes: weights,
    numLayers: model.numLayers,
    usableGpuBytes: usable - overhead - allUsers,
    offload: offloadSpec,
  });
  // Capacity math (max users/context, the context table, the chart) treats the GPU-resident
  // weights as the fixed cost once offload is on, assuming the rest already spilled to RAM —
  // the same simplification planOffload makes, so they agree with the fit/offload badge.
  // Bit-identical to `fixed` when offload is off/absent (no separate computation).
  const fixedBytes = offloadSpec.enabled ? offload.gpuWeightBytes + overhead : fixed;

  const contexts = new Set<number>(TABLE_CONTEXTS);
  contexts.add(ctx);
  const contextTable = [...contexts]
    .sort((a, b) => a - b)
    .map((c) => {
      const kvReq = kvBytesForContext(model, kvContextForRuntime(c, runtime), quant.kv) * kvReplication;
      return { contextTokens: c, kvBytesPerRequest: kvReq, maxUsers: maxUsers(usable, fixedBytes, kvReq) };
    });

  const activeWeightBytes = resolved.activeBytes;
  // Multi-GPU tensor-parallel communication overhead, as a small labelled efficiency
  // penalty per doubling of GPU count (see tensorParallel.ts). 1× at gpuCount 1.
  const throughput = decodeThroughput({
    activeWeightBytes,
    kvBytesPerRequest: perRequest,
    concurrentUsers: users,
    bandwidthGBs: hardware.bandwidthGBs,
    gpuCount: hardware.gpuCount,
    efficiency: DECODE_EFFICIENCY * tensorParallelEfficiency(hardware.gpuCount),
  });

  const pf = prefillFlops({
    activeParams: active.active,
    promptTokens: ctx,
    numLayers: model.numLayers,
    headDim: model.headDim,
    numHeads: model.ffn?.numAttentionHeads,
    hiddenSize: model.hiddenSize,
  });
  const ttft = estimateTtft({ flops: pf.flops, tflopsBf16: hardware.tflopsBf16, gpuCount: hardware.gpuCount });
  // Left untouched (same object, not recomputed) when offload is off, so throughput stays bit-identical.
  const activeSplit = splitByLayerFraction(activeWeightBytes, offload.gpuLayers, model.numLayers);
  const offloadEfficiency = DECODE_EFFICIENCY * tensorParallelEfficiency(hardware.gpuCount);
  const offloadThroughput = !offloadSpec.enabled
    ? throughput
    : !offload.fitsInRam
      // Doesn't actually run (even with RAM): don't present a throughput number as achievable.
      ? { perUserTokS: 0, aggregateTokS: 0, efficiency: offloadEfficiency }
      : offloadDecodeThroughput({
          gpuActiveWeightBytes: activeSplit.gpu,
          cpuActiveWeightBytes: activeSplit.cpu,
          kvBytesPerRequest: perRequest,
          concurrentUsers: users,
          bandwidthGBs: hardware.bandwidthGBs,
          gpuCount: hardware.gpuCount,
          ramBandwidthGBs: offloadSpec.ramBandwidthGBs,
          gpuEfficiency: offloadEfficiency,
        });

  return {
    kvBytesPerToken: perToken,
    kvBytesPerRequest: perRequest,
    kvBytesAllUsers: allUsers,
    weightBytes: weights,
    weightSource: resolved.source,
    activeWeightBytes,
    activeParams: active.active,
    activeParamsMethod: active.method,
    overheadBytes: overhead,
    usableBytes: usable,
    totalBytes: total,
    headroomBytes: usable - total,
    fits: total <= usable,
    fixedBytes,
    bytesPerUser: perRequest,
    maxUsersAtContext: maxUsers(usable, fixedBytes, perRequest),
    maxContextForUsers: maxContext(usable, fixedBytes, users, perToken, model.maxPositionEmbeddings, runtime),
    contextTable,
    throughput: offloadThroughput,
    tensorParallel,
    prefill: { ttftSeconds: ttft.ttftSeconds, flops: pf.flops, mfu: ttft.mfu, headsSource: pf.headsSource },
    offload,
  };
}
