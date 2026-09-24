import { describe, expect, it } from 'vitest';
import {
  kvContextForRuntime,
  llamaCppCacheType,
  llamaCppPerSlotContext,
  pagedKvCacheDtype,
  roundUpToBlock,
  RUNTIME_KEYS,
  RUNTIME_PROFILES,
  VLLM_KV_BLOCK_TOKENS,
} from './runtime';

describe('roundUpToBlock', () => {
  it('rounds up to the next multiple of blockSize', () => {
    expect(roundUpToBlock(1, 16)).toBe(16);
    expect(roundUpToBlock(16, 16)).toBe(16);
    expect(roundUpToBlock(17, 16)).toBe(32);
    expect(roundUpToBlock(8192, 16)).toBe(8192);
    expect(roundUpToBlock(8193, 16)).toBe(8208);
  });

  it('clamps negative input to 0 and passes through a non-positive block size', () => {
    expect(roundUpToBlock(-5, 16)).toBe(0);
    expect(roundUpToBlock(100, 0)).toBe(100);
  });
});

describe('kvContextForRuntime', () => {
  it('rounds up to the vLLM KV block size, and only for vLLM', () => {
    expect(kvContextForRuntime(8193, 'vllm')).toBe(roundUpToBlock(8193, VLLM_KV_BLOCK_TOKENS));
    expect(kvContextForRuntime(8192, 'vllm')).toBe(8192); // already a block multiple
    for (const runtime of ['generic', 'llamacpp', 'sglang', 'mlx'] as const) {
      expect(kvContextForRuntime(8193, runtime)).toBe(8193);
    }
  });
});

describe('llamaCppPerSlotContext', () => {
  it('splits the total context pool evenly across slots, flooring', () => {
    expect(llamaCppPerSlotContext(8192, 4)).toBe(2048);
    expect(llamaCppPerSlotContext(8193, 4)).toBe(2048); // remainder dropped
    expect(llamaCppPerSlotContext(100, 0)).toBe(100); // no slots: treat as a single pool
  });
});

describe('llamaCppCacheType', () => {
  it('maps Headroom KV quant keys to llama.cpp cache types', () => {
    expect(llamaCppCacheType('fp16')).toBe('f16');
    expect(llamaCppCacheType('bf16')).toBe('bf16');
    expect(llamaCppCacheType('fp8')).toBe('q8_0');
    expect(llamaCppCacheType('int8')).toBe('q8_0');
    expect(llamaCppCacheType('int4')).toBe('q4_0');
  });
});

describe('pagedKvCacheDtype', () => {
  it('maps Headroom KV quant keys to vLLM/SGLang --kv-cache-dtype', () => {
    expect(pagedKvCacheDtype('fp16')).toBe('auto');
    expect(pagedKvCacheDtype('bf16')).toBe('auto');
    expect(pagedKvCacheDtype('fp8')).toBe('fp8');
    expect(pagedKvCacheDtype('int8')).toBe('fp8');
    expect(pagedKvCacheDtype('int4')).toBe('fp8');
  });
});

describe('RUNTIME_PROFILES', () => {
  it('has an entry for every runtime key', () => {
    for (const key of RUNTIME_KEYS) {
      expect(RUNTIME_PROFILES[key].key).toBe(key);
      expect(RUNTIME_PROFILES[key].label.length).toBeGreaterThan(0);
    }
  });
});
