// Builds a ready-to-paste launch command for the chosen runtime profile from the current
// calculator state. Pure and framework-free: src/ui/LaunchCommand.tsx only renders the result.
import { calculate } from './fit';
import { resolveOffload } from './offload';
import { formatBytes } from './format';
import { findGpuPreset } from './presets/gpus';
import {
  LLAMACPP_DEFAULT_CACHE_TYPE,
  PAGED_DEFAULT_KV_CACHE_DTYPE,
  RUNTIME_PROFILES,
  SGLANG_MEM_FRACTION_STATIC,
  VLLM_GPU_MEMORY_UTILIZATION,
  llamaCppCacheType,
  llamaCppPerSlotContext,
  pagedKvCacheDtype,
} from './runtime';
import type { CalcState, OffloadPlan } from './types';

export interface LaunchCommand {
  /** null for `generic`, which has no serving runtime to launch. */
  command: string | null;
  /** Caveats specific to the generated command (block quantization, unsupported KV dtypes, etc.). */
  notes: string[];
}

/** Characters that never need shell-quoting: a typical HF repo id, path, or number. */
const SAFE_SHELL_ARG = /^[A-Za-z0-9._/:@=+-]+$/;

/** POSIX single-quotes `value` if it contains anything outside the safe set, so it survives as one shell argument. */
export function shellQuote(value: string): string {
  if (SAFE_SHELL_ARG.test(value)) return value;
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** Rough heuristic: llama.cpp's `-hf` resolves a repo containing GGUF files; a plain safetensors repo id 404s. */
function looksLikeGgufRepo(id: string): boolean {
  return /gguf/i.test(id);
}

/** vLLM/SGLang KV dtypes that aren't the runtime's native/auto type get an approximation note. */
function kvApproximationNote(kv: CalcState['quant']['kv'], runtime: 'vLLM' | 'SGLang'): string | null {
  if (kv === 'int8' || kv === 'int4') {
    return `${runtime} has no ${kv.toUpperCase()} KV cache; using its FP8 KV cache as the closest supported approximation.`;
  }
  return null;
}

/** Flags a runtime/GPU-vendor mismatch (MLX needs Apple silicon; vLLM/SGLang target NVIDIA/AMD). Unknown/custom/"other" GPUs are not checked. */
function hardwareMismatchNote(runtime: CalcState['runtime'], gpuName: string): string | null {
  const vendor = findGpuPreset(gpuName)?.vendor;
  if (!vendor || vendor === 'other') return null;
  if (runtime === 'mlx' && vendor !== 'apple') {
    return `${RUNTIME_PROFILES.mlx.label} runs on Apple unified memory; ${gpuName} is not Apple silicon, so this command will not run as-is.`;
  }
  if ((runtime === 'vllm' || runtime === 'sglang') && vendor === 'apple') {
    return `${RUNTIME_PROFILES[runtime].label} targets NVIDIA/AMD GPUs; ${gpuName} is not a supported backend.`;
  }
  return null;
}

/** vLLM's --cpu-offload-gb is GiB per GPU: the RAM-resident weights split across the tensor-parallel GPUs, rounded up. */
export function vllmCpuOffloadGiB(cpuWeightBytes: number, gpuCount: number): number {
  return Math.ceil(cpuWeightBytes / Math.max(1, gpuCount) / 2 ** 30);
}

const OFFLOAD_DOES_NOT_RUN_NOTE = 'This GPU/RAM split does not run even with CPU/RAM offload (see the verdict above), so this command will not start as-is.';

export function buildLaunchCommand(state: CalcState): LaunchCommand {
  const { model, quant, hardware, workload, runtime = 'generic' } = state;
  const id = model.id || model.name;
  const quotedId = shellQuote(id);
  const C = Math.max(0, Math.floor(workload.contextTokens));
  const N = Math.max(1, Math.floor(workload.concurrentUsers));
  const gpuCount = Math.max(1, Math.floor(hardware.gpuCount));
  const mismatchNote = hardwareMismatchNote(runtime, hardware.gpuName);
  // With CPU/RAM offload on, the layer split comes from the same calculate() the verdict shows.
  const offloadPlan: OffloadPlan | null = resolveOffload(hardware.offload).enabled ? calculate(state).offload : null;
  const speculativeOn = state.speculative?.enabled === true;

  switch (runtime) {
    case 'vllm': {
      const parts = [`vllm serve ${quotedId}`, `--max-model-len ${C}`, `--max-num-seqs ${N}`, `--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION}`];
      const kvDtype = pagedKvCacheDtype(quant.kv);
      if (kvDtype !== PAGED_DEFAULT_KV_CACHE_DTYPE) parts.push(`--kv-cache-dtype ${kvDtype}`);
      if (gpuCount > 1) parts.push(`--tensor-parallel-size ${gpuCount}`);
      const notes = [
        'Starting point, not a guarantee: real usage also depends on --enforce-eager, batch size, and the model.',
      ];
      if (offloadPlan && offloadPlan.cpuWeightBytes > 0) {
        const gib = vllmCpuOffloadGiB(offloadPlan.cpuWeightBytes, gpuCount);
        parts.push(`--cpu-offload-gb ${gib}`);
        notes.push(
          `--cpu-offload-gb is GiB per GPU: the ${formatBytes(offloadPlan.cpuWeightBytes)} of weights that don't fit in VRAM${gpuCount > 1 ? `, split across ${gpuCount} GPUs` : ''}, rounded up. vLLM streams them over PCIe every step, so real speed can differ from the RAM-bandwidth estimate.`,
        );
      }
      if (offloadPlan && !offloadPlan.fitsInRam) notes.push(OFFLOAD_DOES_NOT_RUN_NOTE);
      if (speculativeOn) {
        notes.push("Speculative decoding is not included: configure the draft model separately with vLLM's speculative-decoding config.");
      }
      const kvNote = kvApproximationNote(quant.kv, 'vLLM');
      if (kvNote) notes.push(kvNote);
      if (mismatchNote) notes.push(mismatchNote);
      return { command: parts.join(' '), notes };
    }
    case 'llamacpp': {
      const totalCtx = C * N;
      const perSlot = llamaCppPerSlotContext(totalCtx, N);
      const isGguf = looksLikeGgufRepo(id);
      const parts = [isGguf ? `llama-server -hf ${quotedId}` : 'llama-server -m /path/to/model.gguf', `-c ${totalCtx}`];
      if (N > 1) parts.push(`-np ${N}`);
      // Offload on: the computed GPU layer count (what the verdict shows). Off: every layer on the GPU.
      parts.push(offloadPlan ? `-ngl ${offloadPlan.gpuLayers}` : '-ngl 999');
      const cacheType = llamaCppCacheType(quant.kv);
      if (cacheType !== LLAMACPP_DEFAULT_CACHE_TYPE) parts.push(`--cache-type-k ${cacheType}`, `--cache-type-v ${cacheType}`);
      const notes = [
        offloadPlan
          ? `Starting point, not a guarantee: -ngl ${offloadPlan.gpuLayers} keeps ${offloadPlan.gpuLayers} of ${model.numLayers} layers on the GPU and runs the other ${offloadPlan.cpuLayers} from system RAM.`
          : 'Starting point, not a guarantee: -ngl 999 offloads every layer, which needs enough VRAM for the whole model.',
      ];
      if (offloadPlan && !offloadPlan.fitsInRam) notes.push(OFFLOAD_DOES_NOT_RUN_NOTE);
      if (!isGguf) {
        notes.push(
          `llama.cpp needs a GGUF file: "${id}" does not look like a GGUF repo. Convert it, or find a -GGUF quant of it (e.g. a "…-GGUF" repo) and either replace -m with -hf <that repo> or point -m at a local .gguf file.`,
        );
      }
      if (perSlot < C) notes.push(`Per-slot context (${perSlot} tokens) is below the chosen context (${C} tokens) — raise -c or lower -np.`);
      if (mismatchNote) notes.push(mismatchNote);
      return { command: parts.join(' '), notes };
    }
    case 'sglang': {
      const parts = [
        'python -m sglang.launch_server',
        `--model-path ${quotedId}`,
        `--context-length ${C}`,
        `--max-running-requests ${N}`,
        `--mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}`,
      ];
      const kvDtype = pagedKvCacheDtype(quant.kv);
      if (kvDtype !== PAGED_DEFAULT_KV_CACHE_DTYPE) parts.push(`--kv-cache-dtype ${kvDtype}`);
      if (gpuCount > 1) parts.push(`--tp ${gpuCount}`);
      const notes = ['Starting point, not a guarantee: real usage also depends on batch size and the model.'];
      if (offloadPlan) {
        notes.push('CPU/RAM offload is not modelled for SGLang: this command assumes the whole model fits in VRAM, and has no offload flag.');
      }
      const kvNote = kvApproximationNote(quant.kv, 'SGLang');
      if (kvNote) notes.push(kvNote);
      if (mismatchNote) notes.push(mismatchNote);
      return { command: parts.join(' '), notes };
    }
    case 'mlx': {
      const parts = [`mlx_lm.server --model ${quotedId} --max-kv-size ${C}`];
      const notes = [
        'Starting point, not a guarantee: mlx_lm.server serves one request at a time, so concurrent users share this context sequentially rather than in parallel.',
        'Unified memory is shared with the OS — see the Apple wired-memory limit above.',
      ];
      if (mismatchNote) notes.push(mismatchNote);
      return { command: parts.join(' '), notes };
    }
    case 'generic':
    default:
      return { command: null, notes: [] };
  }
}
