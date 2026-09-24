import type { Attention, FfnSpec, ModelSpec, MoeSpec, NativeDtype } from './types';
import { activeParamsDetailed } from './weights';

type Json = Record<string, unknown>;

const MLA_MODEL_TYPES = new Set(['deepseek_v2', 'deepseek_v3', 'kimi_k2']);

export const HF_ERRORS = {
  gated: 'gated or private: add an HF token, or pick the built-in preset',
  notFound: 'repo not found',
  network: 'could not reach Hugging Face — check your connection and try again',
  emptyId: 'enter a Hugging Face repo id like org/model',
  badConfig: 'config.json could not be read as a model config',
} as const;

function isObject(v: unknown): v is Json {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

function dtypeFrom(v: unknown): NativeDtype | undefined {
  switch (str(v)) {
    case 'bfloat16':
    case 'bf16':
      return 'bf16';
    case 'float16':
    case 'fp16':
    case 'half':
      return 'fp16';
    case 'float32':
    case 'fp32':
    case 'float':
      return 'fp32';
    default:
      return undefined;
  }
}

const HEAD_DIM_MISSING = 'head dimension could not be derived — enter it manually';
const PARAMS_MISSING = 'parameter count not available — enter manually';
export const FP8_WARNING = 'native FP8 weights: FP8 pre-selected; BF16 would double the weight size';
export const VISION_WARNING = 'parameter count includes a vision tower';

const fmtB = (n: number): string => `${Number((n / 1e9).toFixed(2))}B`;

/**
 * Warnings that follow from the spec alone, in a stable order. Regenerated after every
 * Advanced edit so a stale note (MoE turned off, sliding window cleared) disappears.
 */
export function deriveWarnings(spec: ModelSpec): string[] {
  const warnings: string[] = [];
  if (!(spec.headDim > 0)) warnings.push(HEAD_DIM_MISSING);
  const sliding = spec.slidingLayers ?? 0;
  const window = spec.slidingWindow ?? 0;
  if (sliding > 0 && window > 0) {
    warnings.push(
      `sliding-window attention on ${sliding} of ${spec.numLayers} layers (window ${window} tokens): KV for those layers stops growing past the window; some runtimes ignore this and allocate full-context KV`,
    );
  }
  if (spec.attention === 'mla') {
    warnings.push('MLA: KV cache is a single compressed latent per token (kv_lora_rank + qk_rope_head_dim), no separate K and V');
  }
  const moe = spec.moe;
  if (moe && moe.numExperts > 1) {
    const { active, method } = activeParamsDetailed(spec);
    const head = `MoE (${moe.numExperts} experts, ${moe.expertsPerToken} per token${moe.sharedExperts ? `, ${moe.sharedExperts} shared` : ''}): all experts must be resident`;
    warnings.push(
      method === 'structural'
        ? `${head}; ~${fmtB(active)} active per token, a structural estimate from the layer shapes (attention, active experts, LM head)`
        : `${head}; ~${fmtB(active)} active per token from the params × experts ratio, which can understate by a third (attention and embeddings are always active)`,
    );
  }
  if (spec.nativeDtype === 'fp8') warnings.push(FP8_WARNING);
  if (!(spec.params > 0)) warnings.push(PARAMS_MISSING);
  return warnings;
}

// Prefixes of every deriveWarnings() message (including older wordings in shared links).
const DERIVED_PREFIXES = [
  'head dimension could not be derived',
  'sliding-window attention on ',
  'MLA: ',
  'MoE',
  'native FP8 weights',
  'native weights are FP8',
  'parameter count not available',
];

/** Keep config-only notes (vision tower, pre-quantized weights, ...) and regenerate the rest. */
export function refreshWarnings(spec: ModelSpec): string[] {
  const kept = spec.warnings.filter((w) => !DERIVED_PREFIXES.some((p) => w.startsWith(p)));
  return [...deriveWarnings(spec), ...kept];
}

/** Strip whitespace, a leading https://huggingface.co/, and trailing slashes. */
export function normalizeModelId(id: string): string {
  return id
    .trim()
    .replace(/^https?:\/\/(www\.)?huggingface\.co\//i, '')
    .replace(/\/+$/, '');
}

/**
 * Build a ModelSpec from a Hugging Face config.json.
 * Throws when the JSON is not a usable model config (fetchModel catches this).
 */
export function parseConfig(json: unknown, params: number | undefined, id: string): ModelSpec {
  if (!isObject(json)) throw new Error(HF_ERRORS.badConfig);
  const top = json;
  const text = isObject(top.text_config) ? top.text_config : {};
  // Architecture fields live under text_config for multimodal wrappers; fall back to top level.
  const get = (key: string): unknown => text[key] ?? top[key];
  const getNum = (key: string): number | undefined => num(get(key));

  const warnings: string[] = [];
  const modelType = str(text.model_type) ?? str(top.model_type) ?? '';
  const topModelType = str(top.model_type) ?? '';

  const numLayers = getNum('num_hidden_layers') ?? getNum('n_layer') ?? getNum('num_layers');
  const numHeads = getNum('num_attention_heads') ?? getNum('n_head');
  const hiddenSize = getNum('hidden_size') ?? getNum('n_embd') ?? getNum('d_model');
  if (!numLayers || numLayers <= 0) throw new Error(HF_ERRORS.badConfig);

  // Attention: MLA if a DeepSeek-style model or kv_lora_rank is present.
  const kvLoraRank = getNum('kv_lora_rank');
  const attention: Attention =
    MLA_MODEL_TYPES.has(modelType) || MLA_MODEL_TYPES.has(topModelType) || kvLoraRank !== undefined
      ? 'mla'
      : 'mha_gqa';

  const numKvHeads = getNum('num_key_value_heads') ?? numHeads ?? 0;
  let headDim = getNum('head_dim');
  const qkRopeHeadDim = getNum('qk_rope_head_dim');
  if (headDim === undefined && attention === 'mla') {
    const nope = getNum('qk_nope_head_dim');
    if (nope !== undefined && qkRopeHeadDim !== undefined) headDim = nope + qkRopeHeadDim;
  }
  if (headDim === undefined && hiddenSize && numHeads) headDim = hiddenSize / numHeads;
  if (headDim === undefined) headDim = 0;

  // MoE
  const numExperts = getNum('num_local_experts') ?? getNum('n_routed_experts') ?? getNum('num_experts');
  let moe: MoeSpec | undefined;
  if (numExperts !== undefined && numExperts > 1) {
    moe = {
      numExperts,
      expertsPerToken: getNum('num_experts_per_tok') ?? getNum('experts_per_token') ?? 1,
      sharedExperts: getNum('n_shared_experts') ?? 0,
    };
  }

  // Sliding-window layers
  const window = getNum('sliding_window');
  let slidingLayers = 0;
  const layerTypes = get('layer_types');
  const pattern = getNum('sliding_window_pattern');
  if (Array.isArray(layerTypes)) {
    slidingLayers = layerTypes.filter((t) => t === 'sliding_attention').length;
  } else if (pattern !== undefined && pattern > 0) {
    slidingLayers = numLayers - Math.ceil(numLayers / pattern);
  } else if (get('use_sliding_window') === true || (window !== undefined && modelType === 'mistral')) {
    slidingLayers = numLayers;
  }
  const hasSliding = slidingLayers > 0 && window !== undefined && window > 0;

  // Native dtype
  const quantCfg = isObject(get('quantization_config')) ? (get('quantization_config') as Json) : undefined;
  const quantMethod = str(quantCfg?.quant_method)?.toLowerCase();
  let nativeDtype: NativeDtype = dtypeFrom(get('torch_dtype')) ?? dtypeFrom(get('dtype')) ?? 'bf16';
  if (quantMethod === 'fp8') {
    nativeDtype = 'fp8';
  }

  let maxPos = getNum('max_position_embeddings') ?? getNum('n_positions') ?? getNum('max_sequence_length');
  if (maxPos === undefined) {
    maxPos = 4096;
    warnings.push('max_position_embeddings missing — assumed 4096');
  }

  const totalParams = params !== undefined && Number.isFinite(params) && params > 0 ? params : 0;

  // Layer shapes for the structural active-params estimate.
  const intermediate = getNum('intermediate_size') ?? getNum('n_inner') ?? getNum('ffn_dim');
  let ffn: FfnSpec | undefined;
  if (intermediate !== undefined && numHeads !== undefined) {
    ffn = { intermediateSize: intermediate, numAttentionHeads: numHeads, tieEmbeddings: get('tie_word_embeddings') === true };
    const moeIntermediate = getNum('moe_intermediate_size');
    const firstKDense = getNum('first_k_dense_replace');
    const qLoraRank = getNum('q_lora_rank');
    const vHeadDim = getNum('v_head_dim');
    if (moeIntermediate !== undefined) ffn.moeIntermediateSize = moeIntermediate;
    if (firstKDense !== undefined) ffn.firstKDense = firstKDense;
    if (qLoraRank !== undefined) ffn.qLoraRank = qLoraRank;
    if (vHeadDim !== undefined) ffn.vHeadDim = vHeadDim;
  }

  // Config-only notes (the spec-derived ones come from deriveWarnings below).
  if (isObject(top.vision_config) || (isObject(top.text_config) && topModelType !== '' && !topModelType.includes('text'))) {
    warnings.push(VISION_WARNING);
  }
  if (quantMethod === 'mxfp4') {
    warnings.push('weights ship as MXFP4 (~4.25 bits/weight for the experts); pick a 4-bit weight quant to match the download size');
  } else if (quantMethod && quantMethod !== 'fp8') {
    warnings.push(`repo ships pre-quantized weights (${quantMethod}); pick the matching weight quant`);
  }
  const name = id.split('/').pop() || id;
  const spec: ModelSpec = {
    id,
    name,
    params: totalParams,
    activeParams: 0,
    numLayers,
    attention,
    numKvHeads,
    headDim,
    slidingLayers: hasSliding ? slidingLayers : 0,
    maxPositionEmbeddings: maxPos,
    hiddenSize: hiddenSize ?? 0,
    vocabSize: getNum('vocab_size') ?? 0,
    nativeDtype,
    source: 'hf',
    warnings,
  };
  if (attention === 'mla') {
    spec.kvLoraRank = kvLoraRank ?? 0;
    spec.qkRopeHeadDim = qkRopeHeadDim ?? 0;
  }
  if (hasSliding) spec.slidingWindow = window;
  if (moe) spec.moe = moe;
  if (ffn) spec.ffn = ffn;
  spec.activeParams = activeParamsDetailed(spec).active;
  spec.warnings = [...deriveWarnings(spec), ...warnings];
  return spec;
}

/** A repo's quantization_config, when it ships pre-quantized weights. */
export interface PreQuant {
  method: string;
  bits?: number;
}

/** quantization_config.quant_method (lower-cased) and bits, from config.json or its text_config. */
export function preQuantOf(config: unknown): PreQuant | undefined {
  if (!isObject(config)) return undefined;
  const text = isObject(config.text_config) ? config.text_config : {};
  const q = isObject(text.quantization_config) ? text.quantization_config : config.quantization_config;
  if (!isObject(q)) return undefined;
  const method = str(q.quant_method)?.toLowerCase();
  if (!method) return undefined;
  const bits = num(q.bits);
  return bits === undefined ? { method } : { method, bits };
}

export type FetchModelResult = { ok: true; spec: ModelSpec; quant?: PreQuant } | { ok: false; error: string };

function statusError(status: number): string {
  if (status === 401 || status === 403) return HF_ERRORS.gated;
  if (status === 404) return HF_ERRORS.notFound;
  return `Hugging Face returned HTTP ${status}`;
}

/**
 * Fetch config.json and the safetensors parameter count for a Hub repo.
 * Never throws: every failure comes back as { ok: false, error }.
 */
export async function fetchModel(
  rawId: string,
  token?: string,
  fetchImpl: typeof fetch = fetch,
): Promise<FetchModelResult> {
  const id = normalizeModelId(rawId ?? '');
  if (!id || !id.includes('/')) return { ok: false, error: HF_ERRORS.emptyId };

  const headers: Record<string, string> = {};
  if (token && token.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const path = id.split('/').map(encodeURIComponent).join('/');
  const apiUrl = `https://huggingface.co/api/models/${path}?expand[]=safetensors&expand[]=gated`;
  const configUrl = `https://huggingface.co/${path}/resolve/main/config.json`;

  let apiRes: Response;
  let configRes: Response;
  try {
    [apiRes, configRes] = await Promise.all([
      fetchImpl(apiUrl, { headers }),
      fetchImpl(configUrl, { headers }),
    ]);
  } catch {
    return { ok: false, error: HF_ERRORS.network };
  }

  if (!configRes.ok) return { ok: false, error: statusError(configRes.status) };

  let config: unknown;
  try {
    config = await configRes.json();
  } catch {
    return { ok: false, error: HF_ERRORS.badConfig };
  }

  // The param count is best-effort: a failed or partial API response just leaves params missing.
  let params: number | undefined;
  if (apiRes.ok) {
    try {
      const api: unknown = await apiRes.json();
      if (isObject(api) && isObject(api.safetensors)) params = num(api.safetensors.total);
    } catch {
      params = undefined;
    }
  }

  try {
    const spec = parseConfig(config, params, id);
    const quant = preQuantOf(config);
    return quant ? { ok: true, spec, quant } : { ok: true, spec };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : HF_ERRORS.badConfig };
  }
}
