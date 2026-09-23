import type { Attention, ModelSpec, MoeSpec, NativeDtype } from './types';
import { activeParams } from './weights';

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
  if (headDim === undefined) {
    headDim = 0;
    warnings.push('head dimension could not be derived — enter it manually');
  }

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

  // Warnings, in a stable order.
  if (hasSliding) {
    warnings.push(
      `sliding-window attention on ${slidingLayers} of ${numLayers} layers (window ${window} tokens): KV for those layers stops growing past the window; some runtimes ignore this and allocate full-context KV`,
    );
  }
  if (attention === 'mla') {
    warnings.push('MLA: KV cache is a single compressed latent per token (kv_lora_rank + qk_rope_head_dim), no separate K and V');
  }
  if (moe) {
    warnings.push(
      `MoE (${moe.numExperts} experts, ${moe.expertsPerToken} per token${moe.sharedExperts ? `, ${moe.sharedExperts} shared` : ''}): all experts must be resident; active params are an estimate that slightly understates (attention and embeddings are always active)`,
    );
  }
  if (nativeDtype === 'fp8') {
    warnings.push('native weights are FP8; the quant table starts at fp8 for this model');
  }
  if (quantMethod === 'mxfp4') {
    warnings.push('weights ship as MXFP4 (~4.25 bits/weight for the experts); pick a 4-bit weight quant to match the download size');
  } else if (quantMethod && quantMethod !== 'fp8') {
    warnings.push(`repo ships pre-quantized weights (${quantMethod}); pick the matching weight quant`);
  }
  if (totalParams === 0) {
    warnings.push('parameter count not available — enter manually');
  }

  const name = id.split('/').pop() || id;
  const spec: ModelSpec = {
    id,
    name,
    params: totalParams,
    activeParams: activeParams(totalParams, moe),
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
  return spec;
}

export type FetchModelResult = { ok: true; spec: ModelSpec } | { ok: false; error: string };

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
    return { ok: true, spec: parseConfig(config, params, id) };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : HF_ERRORS.badConfig };
  }
}
