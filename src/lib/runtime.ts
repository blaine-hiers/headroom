// Runtime profiles: bundled static data + the pure math each profile changes from the
// `generic` default (today's behaviour). Usable-VRAM/overhead combinators that build on
// fit.ts's existing helpers live in fit.ts itself (usableBytesForRuntime, overheadBytesForRuntime)
// to avoid a circular import; this file stays framework-free and dependency-free.
import type { KvQuantKey, RuntimeKey } from './types';

export interface RuntimeProfile {
  key: RuntimeKey;
  label: string;
  /** One-line description shown next to the selector. */
  blurb: string;
}

export const RUNTIME_KEYS: readonly RuntimeKey[] = ['generic', 'vllm', 'llamacpp', 'sglang', 'mlx'];

export const RUNTIME_PROFILES: Record<RuntimeKey, RuntimeProfile> = {
  generic: {
    key: 'generic',
    label: 'Generic',
    blurb: 'Reserve % + fixed overhead. No launch command.',
  },
  vllm: {
    key: 'vllm',
    label: 'vLLM',
    blurb: 'gpu_memory_utilization × VRAM; KV allocated in 16-token blocks.',
  },
  llamacpp: {
    key: 'llamacpp',
    label: 'llama.cpp / Ollama',
    blurb: '-c is one context pool shared across -np parallel slots.',
  },
  sglang: {
    key: 'sglang',
    label: 'SGLang',
    blurb: 'mem_fraction_static × VRAM.',
  },
  mlx: {
    key: 'mlx',
    label: 'MLX',
    blurb: 'Apple unified memory; uses the wired-memory limit above.',
  },
};

/** vLLM's own default `--gpu-memory-utilization`. */
export const VLLM_GPU_MEMORY_UTILIZATION = 0.9;
/** vLLM's paged KV cache allocates in fixed-size token blocks. */
export const VLLM_KV_BLOCK_TOKENS = 16;
/**
 * Flat allowance for CUDA-graph capture buffers and activation memory that vLLM keeps
 * outside the KV-cache pool. This is a conservative, labelled estimate (not measured per
 * model/batch size) — real usage varies with `--enforce-eager`, batch size and model. Shown
 * in the UI and Show the math so the number is never silently assumed.
 */
export const VLLM_OVERHEAD_ALLOWANCE_GB = 1;

/** SGLang's own default `--mem-fraction-static`. */
export const SGLANG_MEM_FRACTION_STATIC = 0.88;

/** Round `tokens` up to the next multiple of `blockSize` (used for vLLM's paged KV cache). */
export function roundUpToBlock(tokens: number, blockSize: number): number {
  const t = Math.max(0, tokens);
  if (!(blockSize > 0)) return t;
  return Math.ceil(t / blockSize) * blockSize;
}

/**
 * Context length to use when sizing KV memory under a runtime profile. vLLM reserves KV
 * cache in fixed-size blocks, so a request's reservation rounds up to the next block; every
 * other profile sizes KV at the exact requested context.
 */
export function kvContextForRuntime(contextTokens: number, runtime: RuntimeKey): number {
  if (runtime === 'vllm') return roundUpToBlock(contextTokens, VLLM_KV_BLOCK_TOKENS);
  return contextTokens;
}

/** llama.cpp's `-c` is one pool shared across `-np` parallel slots; each slot gets floor(c / np) tokens. */
export function llamaCppPerSlotContext(totalContextTokens: number, parallelSlots: number): number {
  if (!(parallelSlots > 0)) return Math.max(0, totalContextTokens);
  return Math.floor(Math.max(0, totalContextTokens) / parallelSlots);
}

/** llama.cpp's default `--cache-type-k` / `--cache-type-v` (unquantized KV). */
export const LLAMACPP_DEFAULT_CACHE_TYPE = 'f16';

const LLAMACPP_CACHE_TYPE: Record<KvQuantKey, string> = {
  fp16: 'f16',
  bf16: 'bf16',
  // llama.cpp's KV cache types are its own quantization scheme (q8_0/q4_0), not IEEE fp8/int —
  // fp8 and int8 both map to its 8-bit KV type, int4 to its 4-bit KV type.
  fp8: 'q8_0',
  int8: 'q8_0',
  int4: 'q4_0',
};

/** Maps Headroom's KV quant key to llama.cpp's `--cache-type-k`/`--cache-type-v` value. */
export function llamaCppCacheType(kv: KvQuantKey): string {
  return LLAMACPP_CACHE_TYPE[kv];
}

/** vLLM/SGLang's default `--kv-cache-dtype` (auto = the model's native dtype, unquantized). */
export const PAGED_DEFAULT_KV_CACHE_DTYPE = 'auto';

const PAGED_KV_CACHE_DTYPE: Record<KvQuantKey, string> = {
  fp16: 'auto',
  bf16: 'auto',
  fp8: 'fp8',
  // Neither vLLM nor SGLang expose an int8/int4 KV cache; fp8 is the closest supported
  // low-precision option, so int8/int4 selections map to it (flagged as an approximation
  // in the launch-command notes).
  int8: 'fp8',
  int4: 'fp8',
};

/** Maps Headroom's KV quant key to vLLM/SGLang's `--kv-cache-dtype` value. */
export function pagedKvCacheDtype(kv: KvQuantKey): string {
  return PAGED_KV_CACHE_DTYPE[kv];
}
