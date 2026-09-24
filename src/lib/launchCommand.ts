// Builds a ready-to-paste launch command for the chosen runtime profile from the current
// calculator state. Pure and framework-free: src/ui/LaunchCommand.tsx only renders the result.
import {
  LLAMACPP_DEFAULT_CACHE_TYPE,
  PAGED_DEFAULT_KV_CACHE_DTYPE,
  SGLANG_MEM_FRACTION_STATIC,
  VLLM_GPU_MEMORY_UTILIZATION,
  llamaCppCacheType,
  llamaCppPerSlotContext,
  pagedKvCacheDtype,
} from './runtime';
import type { CalcState } from './types';

export interface LaunchCommand {
  /** null for `generic`, which has no serving runtime to launch. */
  command: string | null;
  /** Caveats specific to the generated command (block quantization, unsupported KV dtypes, etc.). */
  notes: string[];
}

/** vLLM/SGLang KV dtypes that aren't the runtime's native/auto type get an approximation note. */
function kvApproximationNote(kv: CalcState['quant']['kv'], runtime: 'vLLM' | 'SGLang'): string | null {
  if (kv === 'int8' || kv === 'int4') {
    return `${runtime} has no ${kv.toUpperCase()} KV cache; using its FP8 KV cache as the closest supported approximation.`;
  }
  return null;
}

export function buildLaunchCommand(state: CalcState): LaunchCommand {
  const { model, quant, hardware, workload, runtime } = state;
  const id = model.id || model.name;
  const C = Math.max(0, Math.floor(workload.contextTokens));
  const N = Math.max(1, Math.floor(workload.concurrentUsers));
  const gpuCount = Math.max(1, Math.floor(hardware.gpuCount));

  switch (runtime) {
    case 'vllm': {
      const parts = [`vllm serve ${id}`, `--max-model-len ${C}`, `--max-num-seqs ${N}`, `--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION}`];
      const kvDtype = pagedKvCacheDtype(quant.kv);
      if (kvDtype !== PAGED_DEFAULT_KV_CACHE_DTYPE) parts.push(`--kv-cache-dtype ${kvDtype}`);
      if (gpuCount > 1) parts.push(`--tensor-parallel-size ${gpuCount}`);
      const notes = [
        'Starting point, not a guarantee: real usage also depends on --enforce-eager, batch size, and the model.',
      ];
      const kvNote = kvApproximationNote(quant.kv, 'vLLM');
      if (kvNote) notes.push(kvNote);
      return { command: parts.join(' '), notes };
    }
    case 'llamacpp': {
      const totalCtx = C * N;
      const perSlot = llamaCppPerSlotContext(totalCtx, N);
      const parts = [`llama-server -hf ${id}`, `-c ${totalCtx}`];
      if (N > 1) parts.push(`-np ${N}`);
      parts.push('-ngl 999');
      const cacheType = llamaCppCacheType(quant.kv);
      if (cacheType !== LLAMACPP_DEFAULT_CACHE_TYPE) parts.push(`--cache-type-k ${cacheType}`, `--cache-type-v ${cacheType}`);
      const notes = ['Starting point, not a guarantee: -ngl 999 offloads every layer, which needs enough VRAM for the whole model.'];
      if (perSlot < C) notes.push(`Per-slot context (${perSlot} tokens) is below the chosen context (${C} tokens) — raise -c or lower -np.`);
      return { command: parts.join(' '), notes };
    }
    case 'sglang': {
      const parts = [
        'python -m sglang.launch_server',
        `--model-path ${id}`,
        `--context-length ${C}`,
        `--max-running-requests ${N}`,
        `--mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}`,
      ];
      const kvDtype = pagedKvCacheDtype(quant.kv);
      if (kvDtype !== PAGED_DEFAULT_KV_CACHE_DTYPE) parts.push(`--kv-cache-dtype ${kvDtype}`);
      if (gpuCount > 1) parts.push(`--tp ${gpuCount}`);
      const notes = ['Starting point, not a guarantee: real usage also depends on batch size and the model.'];
      const kvNote = kvApproximationNote(quant.kv, 'SGLang');
      if (kvNote) notes.push(kvNote);
      return { command: parts.join(' '), notes };
    }
    case 'mlx': {
      const parts = [`mlx_lm.server --model ${id} --max-kv-size ${C}`];
      const notes = [
        'Starting point, not a guarantee: mlx_lm.server serves one request at a time, so concurrent users share this context sequentially rather than in parallel.',
        'Unified memory is shared with the OS — see the Apple wired-memory limit above.',
      ];
      return { command: parts.join(' '), notes };
    }
    case 'generic':
    default:
      return { command: null, notes: [] };
  }
}
