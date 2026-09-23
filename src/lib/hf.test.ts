import { describe, expect, it, vi } from 'vitest';
import deepseekRaw from './__fixtures__/deepseek-v3.json?raw';
import gemmaRaw from './__fixtures__/gemma-3-27b.json?raw';
import gptOssRaw from './__fixtures__/gpt-oss-120b.json?raw';
import llamaRaw from './__fixtures__/llama-3-70b.json?raw';
import mistralRaw from './__fixtures__/mistral-7b-v0.1.json?raw';
import qwenApiRaw from './__fixtures__/qwen2.5-7b-instruct.api.json?raw';
import qwenRaw from './__fixtures__/qwen2.5-7b-instruct.json?raw';
import qwenMoeRaw from './__fixtures__/qwen3-30b-a3b.json?raw';
import { HF_ERRORS, fetchModel, normalizeModelId, parseConfig } from './hf';

const j = (raw: string): unknown => JSON.parse(raw);

describe('parseConfig on real fixtures', () => {
  it('Qwen2.5-7B-Instruct: GQA, use_sliding_window false → no sliding', () => {
    const s = parseConfig(j(qwenRaw), 7_615_616_512, 'Qwen/Qwen2.5-7B-Instruct');
    expect(s).toMatchObject({
      id: 'Qwen/Qwen2.5-7B-Instruct',
      name: 'Qwen2.5-7B-Instruct',
      params: 7_615_616_512,
      activeParams: 7_615_616_512,
      numLayers: 28,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      slidingLayers: 0,
      maxPositionEmbeddings: 32768,
      hiddenSize: 3584,
      vocabSize: 152064,
      nativeDtype: 'bf16',
      source: 'hf',
    });
    expect(s.slidingWindow).toBeUndefined();
    expect(s.moe).toBeUndefined();
    expect(s.warnings).toEqual([]);
  });

  it('Qwen3-30B-A3B: MoE from num_experts', () => {
    const s = parseConfig(j(qwenMoeRaw), 30_532_122_624, 'Qwen/Qwen3-30B-A3B');
    expect(s.numLayers).toBe(48);
    expect(s.numKvHeads).toBe(4);
    expect(s.headDim).toBe(128);
    expect(s.hiddenSize).toBe(2048);
    expect(s.maxPositionEmbeddings).toBe(40960);
    expect(s.moe).toEqual({ numExperts: 128, expertsPerToken: 8, sharedExperts: 0 });
    expect(s.activeParams).toBeCloseTo((30_532_122_624 * 8) / 128, 0);
    expect(s.slidingLayers).toBe(0);
    expect(s.warnings.some((w) => w.startsWith('MoE'))).toBe(true);
  });

  it('DeepSeek-V3: MLA, MoE with shared expert, FP8 native', () => {
    const s = parseConfig(j(deepseekRaw), 671e9, 'deepseek-ai/DeepSeek-V3');
    expect(s.attention).toBe('mla');
    expect(s.kvLoraRank).toBe(512);
    expect(s.qkRopeHeadDim).toBe(64);
    expect(s.headDim).toBe(192);
    expect(s.numLayers).toBe(61);
    expect(s.moe).toEqual({ numExperts: 256, expertsPerToken: 8, sharedExperts: 1 });
    expect(s.nativeDtype).toBe('fp8');
    expect(s.maxPositionEmbeddings).toBe(163840);
    expect(s.warnings.some((w) => w.startsWith('MLA'))).toBe(true);
    expect(s.warnings.some((w) => w.includes('FP8'))).toBe(true);
  });

  it('gpt-oss-120b: layer_types sliding count, MoE via num_local_experts, MXFP4 warning', () => {
    const s = parseConfig(j(gptOssRaw), 116_829_156_672, 'openai/gpt-oss-120b');
    expect(s.numLayers).toBe(36);
    expect(s.numKvHeads).toBe(8);
    expect(s.headDim).toBe(64);
    expect(s.slidingLayers).toBe(18);
    expect(s.slidingWindow).toBe(128);
    expect(s.moe).toEqual({ numExperts: 128, expertsPerToken: 4, sharedExperts: 0 });
    expect(s.nativeDtype).toBe('bf16');
    expect(s.warnings.some((w) => w.includes('MXFP4'))).toBe(true);
    expect(s.warnings.some((w) => w.startsWith('sliding-window'))).toBe(true);
  });

  it('Mistral-7B-v0.1: sliding_window + model_type mistral → all layers sliding', () => {
    const s = parseConfig(j(mistralRaw), 7_241_732_096, 'mistralai/Mistral-7B-v0.1');
    expect(s.slidingLayers).toBe(32);
    expect(s.slidingWindow).toBe(4096);
    expect(s.numKvHeads).toBe(8);
    expect(s.headDim).toBe(128);
  });

  it('Llama 3 70B: GQA, no sliding, bf16', () => {
    const s = parseConfig(j(llamaRaw), 70.6e9, 'meta-llama/Meta-Llama-3-70B');
    expect(s).toMatchObject({ numLayers: 80, numKvHeads: 8, headDim: 128, hiddenSize: 8192, slidingLayers: 0, nativeDtype: 'bf16' });
  });

  it('Gemma 3 27B: reads text_config, sliding_window_pattern 6 → 51 of 62', () => {
    const s = parseConfig(j(gemmaRaw), 27_432_406_640, 'google/gemma-3-27b-it');
    expect(s).toMatchObject({
      numLayers: 62,
      numKvHeads: 16,
      headDim: 128,
      hiddenSize: 5376,
      slidingWindow: 1024,
      slidingLayers: 51,
      maxPositionEmbeddings: 131072,
      vocabSize: 262208,
      nativeDtype: 'bf16',
    });
  });
});

describe('parseConfig edge cases', () => {
  it('missing params → 0 with a warning', () => {
    const s = parseConfig(j(llamaRaw), undefined, 'x/y');
    expect(s.params).toBe(0);
    expect(s.warnings).toContain('parameter count not available — enter manually');
  });

  it('falls back from dtype, marks fp16/fp32, and warns on other pre-quantized repos', () => {
    const base = { num_hidden_layers: 2, num_attention_heads: 4, hidden_size: 256, max_position_embeddings: 2048 };
    expect(parseConfig({ ...base, dtype: 'float16' }, 1, 'a/b').nativeDtype).toBe('fp16');
    expect(parseConfig({ ...base, torch_dtype: 'float32' }, 1, 'a/b').nativeDtype).toBe('fp32');
    const awq = parseConfig({ ...base, quantization_config: { quant_method: 'awq' } }, 1, 'a/b');
    expect(awq.warnings.some((w) => w.includes('awq'))).toBe(true);
  });

  it('use_sliding_window true → all layers sliding', () => {
    const s = parseConfig({ num_hidden_layers: 4, num_attention_heads: 4, hidden_size: 256, sliding_window: 512, use_sliding_window: true }, 1, 'a/b');
    expect(s.slidingLayers).toBe(4);
    expect(s.maxPositionEmbeddings).toBe(4096);
    expect(s.warnings.some((w) => w.includes('max_position_embeddings'))).toBe(true);
  });

  it('kv_lora_rank alone triggers MLA', () => {
    const s = parseConfig({ model_type: 'custom', num_hidden_layers: 2, num_attention_heads: 4, hidden_size: 256, kv_lora_rank: 128, qk_rope_head_dim: 32 }, 1, 'a/b');
    expect(s.attention).toBe('mla');
    expect(s.kvLoraRank).toBe(128);
  });

  it('throws on non-object or layer-less JSON', () => {
    expect(() => parseConfig(null, 1, 'a/b')).toThrow();
    expect(() => parseConfig([1, 2], 1, 'a/b')).toThrow();
    expect(() => parseConfig({ hidden_size: 5 }, 1, 'a/b')).toThrow();
  });
});

describe('normalizeModelId', () => {
  it('trims whitespace and the huggingface.co prefix', () => {
    expect(normalizeModelId('  Qwen/Qwen3-32B ')).toBe('Qwen/Qwen3-32B');
    expect(normalizeModelId('https://huggingface.co/Qwen/Qwen3-32B/')).toBe('Qwen/Qwen3-32B');
  });
});

function res(status: number, body: unknown): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status });
}

describe('fetchModel', () => {
  it('200: combines config.json and safetensors.total, sends the token', async () => {
    const fetchImpl = vi.fn(async (url: string | URL | Request) =>
      String(url).includes('/api/models/') ? res(200, j(qwenApiRaw)) : res(200, j(qwenRaw)),
    );
    const r = await fetchModel(' https://huggingface.co/Qwen/Qwen2.5-7B-Instruct ', 'hf_abc', fetchImpl as typeof fetch);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.spec.id).toBe('Qwen/Qwen2.5-7B-Instruct');
    expect(r.spec.params).toBe(7_615_616_512);
    const urls = fetchImpl.mock.calls.map((c) => String(c[0]));
    expect(urls).toContain('https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct?expand[]=safetensors&expand[]=gated');
    expect(urls).toContain('https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json');
    const init = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect((init[1].headers as Record<string, string>).Authorization).toBe('Bearer hf_abc');
  });

  it('200 config but failed API → params missing warning, still ok', async () => {
    const fetchImpl = async (url: string | URL | Request) =>
      String(url).includes('/api/models/') ? res(500, 'err') : res(200, j(qwenRaw));
    const r = await fetchModel('Qwen/Qwen2.5-7B-Instruct', undefined, fetchImpl as typeof fetch);
    expect(r.ok && r.spec.params === 0 && r.spec.warnings.includes('parameter count not available — enter manually')).toBe(true);
  });

  it('401 → gated message', async () => {
    const r = await fetchModel('meta-llama/Llama-3.1-8B', undefined, (async () => res(401, 'no')) as typeof fetch);
    expect(r).toEqual({ ok: false, error: HF_ERRORS.gated });
  });

  it('403 → gated message', async () => {
    const r = await fetchModel('a/b', undefined, (async () => res(403, 'no')) as typeof fetch);
    expect(r).toEqual({ ok: false, error: HF_ERRORS.gated });
  });

  it('404 → repo not found', async () => {
    const r = await fetchModel('a/b', undefined, (async () => res(404, 'no')) as typeof fetch);
    expect(r).toEqual({ ok: false, error: 'repo not found' });
  });

  it('other status → HTTP code message', async () => {
    const r = await fetchModel('a/b', undefined, (async () => res(503, 'no')) as typeof fetch);
    expect(r).toEqual({ ok: false, error: 'Hugging Face returned HTTP 503' });
  });

  it('thrown network error → plain message, never throws', async () => {
    const r = await fetchModel('a/b', undefined, (async () => {
      throw new TypeError('Failed to fetch');
    }) as typeof fetch);
    expect(r).toEqual({ ok: false, error: HF_ERRORS.network });
  });

  it('invalid JSON or unusable config → error result', async () => {
    const bad = await fetchModel('a/b', undefined, (async () => res(200, 'not json{')) as typeof fetch);
    expect(bad).toEqual({ ok: false, error: HF_ERRORS.badConfig });
    const empty = await fetchModel('a/b', undefined, (async () => res(200, {})) as typeof fetch);
    expect(empty).toEqual({ ok: false, error: HF_ERRORS.badConfig });
  });

  it('empty or malformed id → error without fetching', async () => {
    const fetchImpl = vi.fn();
    expect(await fetchModel('   ', undefined, fetchImpl as unknown as typeof fetch)).toEqual({ ok: false, error: HF_ERRORS.emptyId });
    expect(await fetchModel('justaname', undefined, fetchImpl as unknown as typeof fetch)).toEqual({ ok: false, error: HF_ERRORS.emptyId });
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
