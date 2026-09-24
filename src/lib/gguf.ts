import { parseConfig } from './hf';
import type { ModelSpec } from './types';

/** A scalar GGUF metadata value. 64-bit integers become plain numbers. */
export type GgufValue = number | string | boolean;

export interface GgufHeader {
  version: number;
  tensorCount: number;
  kvCount: number;
  /** Scalar key-values. Array values are skipped; their lengths are in `arrayLengths`. */
  metadata: Record<string, GgufValue>;
  arrayLengths: Record<string, number>;
  /** False when the buffer ended before every key-value was read (a ranged read of a big header). */
  complete: boolean;
}

export const GGUF_ERRORS = {
  notGguf: 'not a GGUF file',
  version: 'unsupported GGUF version (only v2 and v3 are read)',
  noArch: 'the GGUF header has no general.architecture',
  noLayers: 'the GGUF header has no block_count for its architecture',
} as const;

// GGUF value type ids.
const T = { U8: 0, I8: 1, U16: 2, I16: 3, U32: 4, I32: 5, F32: 6, BOOL: 7, STRING: 8, ARRAY: 9, U64: 10, I64: 11, F64: 12 } as const;
const FIXED_SIZE: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

/** Thrown internally when a value runs past the end of the buffer. */
class Truncated extends Error {}

/**
 * Read the header and metadata key-values of a GGUF (v2 or v3, either byte order).
 * Works on a prefix of the file: it stops cleanly at the end of the buffer and sets
 * `complete: false`. Throws only when the buffer is not a GGUF at all.
 */
export function parseGgufHeader(buf: ArrayBuffer): GgufHeader {
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  if (buf.byteLength < 24 || view.getUint32(0, true) !== 0x46554747) throw new Error(GGUF_ERRORS.notGguf); // "GGUF"
  // Byte order: a big-endian file's version reads as e.g. 0x03000000 in little-endian.
  let le = true;
  let version = view.getUint32(4, true);
  if (version > 0xffff) {
    le = false;
    version = view.getUint32(4, false);
  }
  if (version !== 2 && version !== 3) throw new Error(GGUF_ERRORS.version);

  const decoder = new TextDecoder();
  let p = 8;
  const need = (n: number) => {
    if (p + n > buf.byteLength) throw new Truncated();
  };
  const u32 = () => {
    need(4);
    const v = view.getUint32(p, le);
    p += 4;
    return v;
  };
  const u64 = () => {
    need(8);
    const v = Number(view.getBigUint64(p, le));
    p += 8;
    return v;
  };
  const string = () => {
    const n = u64();
    need(n);
    const s = decoder.decode(bytes.subarray(p, p + n));
    p += n;
    return s;
  };
  const scalar = (type: number): GgufValue => {
    const size = FIXED_SIZE[type];
    if (size === undefined) throw new Error(`unknown GGUF value type ${type}`);
    need(size);
    let v: GgufValue;
    switch (type) {
      case T.U8: v = view.getUint8(p); break;
      case T.I8: v = view.getInt8(p); break;
      case T.U16: v = view.getUint16(p, le); break;
      case T.I16: v = view.getInt16(p, le); break;
      case T.U32: v = view.getUint32(p, le); break;
      case T.I32: v = view.getInt32(p, le); break;
      case T.F32: v = view.getFloat32(p, le); break;
      case T.BOOL: v = view.getUint8(p) !== 0; break;
      case T.U64: v = Number(view.getBigUint64(p, le)); break;
      case T.I64: v = Number(view.getBigInt64(p, le)); break;
      default: v = view.getFloat64(p, le);
    }
    p += size;
    return v;
  };
  const skipArray = (type: number, len: number) => {
    if (type === T.STRING) {
      for (let i = 0; i < len; i++) {
        const n = u64();
        need(n);
        p += n;
      }
    } else if (type === T.ARRAY) {
      for (let i = 0; i < len; i++) skipArray(u32(), u64());
    } else {
      const size = FIXED_SIZE[type];
      if (size === undefined) throw new Error(`unknown GGUF value type ${type}`);
      need(size * len);
      p += size * len;
    }
  };

  const header: GgufHeader = { version, tensorCount: 0, kvCount: 0, metadata: {}, arrayLengths: {}, complete: false };
  header.tensorCount = u64();
  header.kvCount = u64();
  try {
    for (let i = 0; i < header.kvCount; i++) {
      const key = string();
      const type = u32();
      if (type === T.STRING) {
        header.metadata[key] = string();
      } else if (type === T.ARRAY) {
        const itemType = u32();
        const len = u64();
        header.arrayLengths[key] = len; // recorded first, so a truncated array still reports its length
        skipArray(itemType, len);
      } else {
        header.metadata[key] = scalar(type);
      }
    }
    header.complete = true;
  } catch (e) {
    if (!(e instanceof Truncated)) throw e;
  }
  return header;
}

/**
 * Architectures whose sliding-window layer pattern llama.cpp hard-codes rather than
 * storing in the file: every Nth layer is full attention, the rest slide.
 */
const SLIDING_PATTERN: Record<string, number> = { gemma2: 2, gemma3: 6, 'gpt-oss': 2, cohere2: 4 };

/**
 * Map GGUF metadata onto the config.json field names parseConfig reads, so both paths
 * share one set of derivation rules.
 */
export function ggufToConfig(header: GgufHeader): Record<string, unknown> {
  const md = header.metadata;
  const arch = md['general.architecture'];
  if (typeof arch !== 'string' || !arch) throw new Error(GGUF_ERRORS.noArch);
  const k = (key: string): number | undefined => {
    const v = md[`${arch}.${key}`];
    return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
  };
  if (k('block_count') === undefined) throw new Error(GGUF_ERRORS.noLayers);
  const kvLoraRank = k('attention.kv_lora_rank');
  const mla = kvLoraRank !== undefined;
  const window = k('attention.sliding_window');
  const config: Record<string, unknown> = {
    model_type: arch,
    num_hidden_layers: k('block_count'),
    num_attention_heads: k('attention.head_count'),
    num_key_value_heads: k('attention.head_count_kv'),
    // MLA files converted for llama.cpp's MQA path store the per-head qk size as key_length_mla.
    head_dim: k('attention.key_length_mla') ?? k('attention.key_length'),
    hidden_size: k('embedding_length'),
    max_position_embeddings: k('context_length'),
    vocab_size: k('vocab_size') ?? header.arrayLengths['tokenizer.ggml.tokens'],
    intermediate_size: k('feed_forward_length'),
    moe_intermediate_size: k('expert_feed_forward_length'),
    num_local_experts: k('expert_count'),
    num_experts_per_tok: k('expert_used_count'),
    n_shared_experts: k('expert_shared_count'),
    first_k_dense_replace: k('leading_dense_block_count'),
  };
  if (mla) {
    config.kv_lora_rank = kvLoraRank;
    config.qk_rope_head_dim = k('rope.dimension_count');
    config.q_lora_rank = k('attention.q_lora_rank');
    config.v_head_dim = k('attention.value_length_mla') ?? k('attention.value_length');
  }
  if (window !== undefined && window > 0) {
    config.sliding_window = window;
    config.sliding_window_pattern = k('attention.sliding_window_pattern') ?? SLIDING_PATTERN[arch];
  }
  return config;
}

/** ModelSpec from a GGUF header. Throws when the header lacks the architecture. */
export function parseGgufSpec(header: GgufHeader, params: number | undefined, id: string): ModelSpec {
  return parseConfig(ggufToConfig(header), params, id);
}
