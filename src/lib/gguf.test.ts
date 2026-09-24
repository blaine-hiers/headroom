import { describe, expect, it } from 'vitest';
import deepseekB64 from './__fixtures__/deepseek-v3.1-q2_k-00001-of-00005.gguf.b64?raw';
import gemmaB64 from './__fixtures__/gemma-3-27b-it-q4_k_m.gguf.b64?raw';
import gptOssB64 from './__fixtures__/gpt-oss-20b-q4_k_m.gguf.b64?raw';
import qwenMoeB64 from './__fixtures__/qwen3-30b-a3b-q4_k_m.gguf.b64?raw';
import { fromBase64 } from './__fixtures__/base64';
import { GGUF_ERRORS, ggufToConfig, parseGgufHeader, parseGgufSpec } from './gguf';
import { kvBytesPerToken } from './kvcache';

// Fixtures: the first 4 KB of real GGUF files from the Hub, base64-encoded. Each one ends
// inside the tokenizer arrays, like a ranged read of the file head does.

type Kv = [key: string, type: number, value: number | string | boolean | bigint | Array<number | string>, itemType?: number];

/** A synthetic GGUF header (tensor infos omitted: the parser stops after the key-values). */
function buildGguf(kvs: Kv[], { version = 3, le = true } = {}): ArrayBuffer {
  const parts: number[] = [];
  const push = (size: number, write: (v: DataView) => void) => {
    const b = new DataView(new ArrayBuffer(size));
    write(b);
    parts.push(...new Uint8Array(b.buffer));
  };
  const u32 = (v: number) => push(4, (d) => d.setUint32(0, v, le));
  const u64 = (v: number | bigint) => push(8, (d) => d.setBigUint64(0, BigInt(v), le));
  const str = (s: string) => {
    const b = new TextEncoder().encode(s);
    u64(b.length);
    parts.push(...b);
  };
  const value = (type: number, v: unknown) => {
    switch (type) {
      case 0: return push(1, (d) => d.setUint8(0, v as number));
      case 1: return push(1, (d) => d.setInt8(0, v as number));
      case 2: return push(2, (d) => d.setUint16(0, v as number, le));
      case 3: return push(2, (d) => d.setInt16(0, v as number, le));
      case 4: return u32(v as number);
      case 5: return push(4, (d) => d.setInt32(0, v as number, le));
      case 6: return push(4, (d) => d.setFloat32(0, v as number, le));
      case 7: return push(1, (d) => d.setUint8(0, v ? 1 : 0));
      case 8: return str(v as string);
      case 10: return u64(v as number | bigint);
      case 11: return push(8, (d) => d.setBigInt64(0, BigInt(v as number), le));
      case 12: return push(8, (d) => d.setFloat64(0, v as number, le));
    }
  };
  parts.push(0x47, 0x47, 0x55, 0x46); // "GGUF"
  u32(version);
  u64(0);
  u64(kvs.length);
  for (const [key, type, v, itemType] of kvs) {
    str(key);
    u32(type);
    if (type === 9) {
      const items = v as unknown[];
      u32(itemType ?? 4);
      u64(items.length);
      for (const item of items) value(itemType ?? 4, item);
    } else value(type, v);
  }
  return new Uint8Array(parts).buffer;
}

describe('parseGgufHeader', () => {
  it('reads every scalar type and skips arrays, recording their lengths (v3)', () => {
    const buf = buildGguf([
      ['general.architecture', 8, 'llama'],
      ['u8', 0, 200],
      ['i8', 1, -5],
      ['u16', 2, 60000],
      ['i16', 3, -300],
      ['u32', 4, 4_000_000_000],
      ['i32', 5, -7],
      ['f32', 6, 0.5],
      ['bool', 7, true],
      ['u64', 10, 131072n],
      ['i64', 11, -9],
      ['f64', 12, 1e-6],
      ['tokenizer.ggml.tokens', 9, ['a', 'bc', 'def'], 8],
      ['llama.ints', 9, [1, 2, 3, 4], 5],
      ['after', 4, 42],
    ]);
    const h = parseGgufHeader(buf);
    expect(h.version).toBe(3);
    expect(h.complete).toBe(true);
    expect(h.kvCount).toBe(15);
    expect(h.metadata).toEqual({
      'general.architecture': 'llama',
      u8: 200,
      i8: -5,
      u16: 60000,
      i16: -300,
      u32: 4_000_000_000,
      i32: -7,
      f32: 0.5,
      bool: true,
      u64: 131072,
      i64: -9,
      f64: 1e-6,
      after: 42,
    });
    expect(h.arrayLengths).toEqual({ 'tokenizer.ggml.tokens': 3, 'llama.ints': 4 });
  });

  it('reads v2 and big-endian files', () => {
    const kvs: Kv[] = [
      ['general.architecture', 8, 'llama'],
      ['llama.block_count', 4, 32],
    ];
    expect(parseGgufHeader(buildGguf(kvs, { version: 2 })).metadata['llama.block_count']).toBe(32);
    const be = parseGgufHeader(buildGguf(kvs, { le: false }));
    expect(be.version).toBe(3);
    expect(be.metadata['llama.block_count']).toBe(32);
  });

  it('stops cleanly at the end of a truncated buffer', () => {
    const full = buildGguf([
      ['general.architecture', 8, 'llama'],
      ['llama.block_count', 4, 32],
      ['tokenizer.ggml.tokens', 9, ['aaaa', 'bbbb', 'cccc'], 8],
    ]);
    const h = parseGgufHeader(full.slice(0, full.byteLength - 6));
    expect(h.complete).toBe(false);
    expect(h.metadata['llama.block_count']).toBe(32);
    expect(h.arrayLengths['tokenizer.ggml.tokens']).toBe(3);
  });

  it('throws on a non-GGUF buffer or an unsupported version', () => {
    expect(() => parseGgufHeader(new TextEncoder().encode('{"not":"gguf","padding":true}').buffer)).toThrow(GGUF_ERRORS.notGguf);
    expect(() => parseGgufHeader(new ArrayBuffer(4))).toThrow(GGUF_ERRORS.notGguf);
    expect(() => parseGgufHeader(buildGguf([], { version: 1 }))).toThrow(GGUF_ERRORS.version);
  });
});

describe('GGUF headers from the Hub (4 KB fixtures)', () => {
  it('Qwen3-30B-A3B: MoE shapes and the vocab from the tokenizer array length', () => {
    const h = parseGgufHeader(fromBase64(qwenMoeB64));
    expect(h.complete).toBe(false);
    expect(h.metadata['general.architecture']).toBe('qwen3moe');
    const s = parseGgufSpec(h, 30_532_122_624, 'bartowski/Qwen_Qwen3-30B-A3B-GGUF');
    expect(s).toMatchObject({
      numLayers: 48,
      attention: 'mha_gqa',
      numKvHeads: 4,
      headDim: 128,
      hiddenSize: 2048,
      vocabSize: 151936,
      maxPositionEmbeddings: 32768,
      moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 },
      ffn: { intermediateSize: 6144, moeIntermediateSize: 768, numAttentionHeads: 32, tieEmbeddings: false },
      source: 'hf',
    });
    // Same structural estimate as the config.json path.
    expect(s.activeParams / 1e9).toBeCloseTo(3.04, 2);
  });

  it('Gemma 3 27B: sliding window with the hard-coded 5-of-6 layer pattern', () => {
    const s = parseGgufSpec(parseGgufHeader(fromBase64(gemmaB64)), 27e9, 'bartowski/google_gemma-3-27b-it-GGUF');
    expect(s.numLayers).toBe(62);
    expect(s.numKvHeads).toBe(16);
    expect(s.headDim).toBe(128);
    expect(s.slidingWindow).toBe(1024);
    expect(s.slidingLayers).toBe(62 - Math.ceil(62 / 6));
    expect(s.maxPositionEmbeddings).toBe(131072);
  });

  it('gpt-oss-20b: alternating sliding layers and MoE', () => {
    const s = parseGgufSpec(parseGgufHeader(fromBase64(gptOssB64)), 20_914_757_184, 'bartowski/openai_gpt-oss-20b-GGUF');
    expect(s).toMatchObject({ numLayers: 24, numKvHeads: 8, headDim: 64, slidingWindow: 128, slidingLayers: 12 });
    expect(s.moe).toEqual({ numExperts: 32, expertsPerToken: 4, sharedExperts: 0 });
  });

  it('DeepSeek-V3.1 (first shard of a split file): MLA from the *_mla keys', () => {
    const h = parseGgufHeader(fromBase64(deepseekB64));
    const c = ggufToConfig(h);
    expect(c.head_dim).toBe(192); // key_length_mla, not the 576 MQA-form key_length
    const s = parseGgufSpec(h, 671_026_419_200, 'unsloth/DeepSeek-V3.1-GGUF');
    expect(s).toMatchObject({ attention: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, headDim: 192, numLayers: 61, vocabSize: 129280 });
    expect(s.moe).toEqual({ numExperts: 256, expertsPerToken: 8, sharedExperts: 1 });
    expect(s.ffn).toMatchObject({ firstKDense: 3, qLoraRank: 1536, vHeadDim: 128, moeIntermediateSize: 2048 });
    expect(kvBytesPerToken(s, 'bf16')).toBe(61 * (512 + 64) * 2);
  });

  it('a header without the architecture keys throws a GGUF-specific message', () => {
    expect(() => ggufToConfig(parseGgufHeader(buildGguf([['general.name', 8, 'x']])))).toThrow(GGUF_ERRORS.noArch);
    expect(() => ggufToConfig(parseGgufHeader(buildGguf([['general.architecture', 8, 'llama']])))).toThrow(GGUF_ERRORS.noLayers);
  });
});
