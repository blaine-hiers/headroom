import { describe, expect, it } from 'vitest';
import deepseekRaw from './__fixtures__/deepseek-v3.json?raw';
import gemmaRaw from './__fixtures__/gemma-3-27b.json?raw';
import gptOssRaw from './__fixtures__/gpt-oss-120b.json?raw';
import llamaRaw from './__fixtures__/llama-3-70b.json?raw';
import mistralRaw from './__fixtures__/mistral-7b-v0.1.json?raw';
import { makeSpec } from './__fixtures__/makeSpec';
import { formatBytes, formatBytesBinary } from './format';
import { parseConfig } from './hf';
import { effectiveSlidingLayers, kvBytesForContext, kvBytesPerToken, kvBytesPerTokenPerLayer } from './kvcache';
import { weightBytes } from './weights';

const llama70 = parseConfig(JSON.parse(llamaRaw), 70.6e9, 'meta-llama/Meta-Llama-3-70B');

describe('golden: Llama 3 70B, BF16 weights, fp16 KV', () => {
  it('KV per token = 327,680 B (328 KB / 320 KiB)', () => {
    expect(kvBytesPerToken(llama70, 'fp16')).toBe(327_680);
    expect(formatBytes(327_680)).toBe('328 KB');
    expect(formatBytesBinary(327_680)).toBe('320 KiB');
  });

  it('2K context = 671,088,640 B (671 MB / 640 MiB)', () => {
    const b = kvBytesForContext(llama70, 2048, 'fp16');
    expect(b).toBe(671_088_640);
    expect(formatBytes(b)).toBe('671 MB');
    expect(formatBytesBinary(b)).toBe('640 MiB');
  });

  it('8K context = 2.68 GB / 2.5 GiB', () => {
    const b = kvBytesForContext(llama70, 8192, 'fp16');
    expect(b).toBe(2_684_354_560);
    expect(formatBytes(b)).toBe('2.68 GB');
    expect(formatBytesBinary(b)).toBe('2.5 GiB');
  });

  it('32K context = 10.7 GB / 10 GiB', () => {
    const b = kvBytesForContext(llama70, 32768, 'fp16');
    expect(b).toBe(10_737_418_240);
    expect(formatBytes(b)).toBe('10.7 GB');
    expect(formatBytesBinary(b)).toBe('10 GiB');
  });

  it('128K context = 42.9 GB / 40 GiB', () => {
    const b = kvBytesForContext(llama70, 131072, 'fp16');
    expect(b).toBe(42_949_672_960);
    expect(formatBytes(b)).toBe('42.9 GB');
    expect(formatBytesBinary(b)).toBe('40 GiB');
  });

  it('weights 70.6B × 2 ≈ 141 GB', () => {
    const w = weightBytes(70.6e9, 'bf16');
    expect(w).toBe(141.2e9);
    expect(formatBytes(w)).toBe('141 GB');
  });

  it('10 users @ 128K ≈ 429 GB of KV', () => {
    const b = 10 * kvBytesForContext(llama70, 131072, 'fp16');
    expect(b).toBe(429_496_729_600);
    expect(formatBytes(b)).toBe('429 GB');
    expect(formatBytesBinary(b)).toBe('400 GiB');
  });
});

describe('KV quant scaling', () => {
  it('fp8 halves and int4 quarters the fp16 cache', () => {
    expect(kvBytesPerToken(llama70, 'bf16')).toBe(327_680);
    expect(kvBytesPerToken(llama70, 'fp8')).toBe(163_840);
    expect(kvBytesPerToken(llama70, 'int8')).toBe(163_840);
    expect(kvBytesPerToken(llama70, 'int4')).toBe(81_920);
  });
});

describe('MHA vs GQA', () => {
  it('MHA config (no num_key_value_heads) uses num_attention_heads and hidden/heads', () => {
    const cfg = { model_type: 'llama', num_hidden_layers: 32, num_attention_heads: 32, hidden_size: 4096, max_position_embeddings: 4096 };
    const spec = parseConfig(cfg, 6.7e9, 'x/llama-2-7b');
    expect(spec.numKvHeads).toBe(32);
    expect(spec.headDim).toBe(128);
    expect(kvBytesPerToken(spec, 'fp16')).toBe(2 * 32 * 32 * 128 * 2);
  });

  it('GQA is smaller than MHA by heads / kvHeads', () => {
    const mha = makeSpec({ numKvHeads: 64 });
    expect(kvBytesPerToken(mha, 'fp16') / kvBytesPerToken(llama70, 'fp16')).toBe(8);
  });
});

describe('MLA', () => {
  it('DeepSeek-V3 = 61 × (512 + 64) × 2 bytes per token', () => {
    const spec = parseConfig(JSON.parse(deepseekRaw), 671e9, 'deepseek-ai/DeepSeek-V3');
    expect(spec.attention).toBe('mla');
    expect(kvBytesPerTokenPerLayer(spec, 'fp16')).toBe(576 * 2);
    expect(kvBytesPerToken(spec, 'fp16')).toBe(61 * (512 + 64) * 2);
    expect(kvBytesForContext(spec, 1000, 'fp16')).toBe(61 * 576 * 2 * 1000);
  });

  it('MLA with missing latent dims counts as zero rather than NaN', () => {
    const spec = makeSpec({ attention: 'mla' });
    expect(kvBytesPerToken(spec, 'fp16')).toBe(0);
  });
});

describe('sliding window', () => {
  it('Mistral-7B at 32768 counts 4096 per layer', () => {
    const spec = parseConfig(JSON.parse(mistralRaw), 7.24e9, 'mistralai/Mistral-7B-v0.1');
    expect(spec.slidingLayers).toBe(32);
    expect(spec.slidingWindow).toBe(4096);
    const perLayer = 2 * 8 * 128 * 2;
    expect(kvBytesForContext(spec, 32768, 'fp16')).toBe(32 * perLayer * 4096);
    // below the window it grows linearly
    expect(kvBytesForContext(spec, 1000, 'fp16')).toBe(32 * perLayer * 1000);
    // full-attention rate ignores the window
    expect(kvBytesPerToken(spec, 'fp16')).toBe(32 * perLayer);
  });

  it('Gemma 3 27B at 131072 → 11 full layers × 131072 + 51 × 1024', () => {
    const spec = parseConfig(JSON.parse(gemmaRaw), 27.4e9, 'google/gemma-3-27b-it');
    const perLayer = 2 * 16 * 128 * 2;
    expect(spec.slidingLayers).toBe(51);
    expect(kvBytesForContext(spec, 131072, 'fp16')).toBe(perLayer * (11 * 131072 + 51 * 1024));
  });

  it('gpt-oss-120b layer_types → 18 sliding layers of 36 with a 128 window', () => {
    const spec = parseConfig(JSON.parse(gptOssRaw), 116.8e9, 'openai/gpt-oss-120b');
    const perLayer = 2 * 8 * 64 * 2;
    expect(spec.slidingLayers).toBe(18);
    expect(kvBytesForContext(spec, 131072, 'fp16')).toBe(perLayer * (18 * 131072 + 18 * 128));
  });

  it('ignores slidingLayers when no window is set, and clamps to numLayers', () => {
    expect(effectiveSlidingLayers(makeSpec({ slidingLayers: 10 }))).toBe(0);
    expect(effectiveSlidingLayers(makeSpec({ slidingLayers: 999, slidingWindow: 10 }))).toBe(80);
    expect(kvBytesForContext(makeSpec(), -5, 'fp16')).toBe(0);
  });
});
