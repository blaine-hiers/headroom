import { closestWeightQuant, effectiveBitsPerWeight } from './fileWeights';
import { parseGgufHeader, parseGgufSpec } from './gguf';
import { fetchModel, HF_ERRORS, normalizeModelId } from './hf';
import { WEIGHT_QUANTS } from './quant';
import type { PreQuant } from './hf';
import type { FileWeights, ModelSpec, WeightQuantKey } from './types';

/** How much of a GGUF to read: the architecture keys sit in the first few KB, before the tokenizer. */
export const GGUF_HEADER_BYTES = 1 << 20;

export interface RepoFile {
  path: string;
  size: number;
}

/** One pickable GGUF: a single file, or the shards of a split file taken together. */
export interface GgufOption {
  /** Path of the file (the first shard for a split file); its header holds the metadata. */
  path: string;
  /** Quant from the filename (e.g. "Q4_K_M"), else the file name. */
  label: string;
  /** Summed size of every shard. */
  bytes: number;
  shards: number;
}

export interface RepoListing {
  files: RepoFile[];
  /** Parameter count the Hub reads from GGUF files (`gguf.total`). */
  ggufParams?: number;
}

type Json = Record<string, unknown>;
const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const posNum = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : undefined);

/** Files and sizes from a `/api/models/{id}?blobs=true&expand[]=siblings&expand[]=gguf` response. */
export function parseRepoListing(api: unknown): RepoListing {
  const files: RepoFile[] = [];
  if (!isObject(api)) return { files };
  if (Array.isArray(api.siblings)) {
    for (const s of api.siblings) {
      if (!isObject(s) || typeof s.rfilename !== 'string') continue;
      const size = posNum(s.size) ?? (isObject(s.lfs) ? posNum(s.lfs.size) : undefined);
      if (size !== undefined) files.push({ path: s.rfilename, size });
    }
  }
  const ggufParams = isObject(api.gguf) ? posNum(api.gguf.total) : undefined;
  return ggufParams === undefined ? { files } : { files, ggufParams };
}

const SHARD_RE = /-(\d{5})-of-(\d{5})\.gguf$/i;
// llama.cpp quant tags: Q4_K_M, IQ4_XS, Q8_0, BF16, F16, MXFP4, TQ1_0...
const QUANT_RE = /(?:^|[-_.])(I?Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4|TQ\d_\d)(?=$|[-_.])/gi;

/** The quant tag in a GGUF filename, upper-cased: "…-Q4_K_M-00001-of-00002.gguf" → "Q4_K_M". */
export function quantFromFilename(path: string): string | undefined {
  const base = (path.split('/').pop() ?? path).replace(SHARD_RE, '').replace(/\.gguf$/i, '');
  const matches = [...base.matchAll(QUANT_RE)];
  return matches.length ? matches[matches.length - 1][1].toUpperCase() : undefined;
}

/** GGUF weight files, split shards grouped, smallest first. Projector and imatrix files are left out. */
export function ggufOptions(files: RepoFile[]): GgufOption[] {
  const groups = new Map<string, { first: string; firstIndex: number; bytes: number; shards: number }>();
  for (const f of files) {
    if (!/\.gguf$/i.test(f.path)) continue;
    const name = f.path.split('/').pop() ?? f.path;
    if (/mmproj|imatrix/i.test(name)) continue;
    const shard = SHARD_RE.exec(f.path);
    const key = shard ? f.path.replace(SHARD_RE, '.gguf') : f.path;
    const index = shard ? Number(shard[1]) : 1;
    const g = groups.get(key);
    if (!g) groups.set(key, { first: f.path, firstIndex: index, bytes: f.size, shards: 1 });
    else {
      g.bytes += f.size;
      g.shards += 1;
      if (index < g.firstIndex) {
        g.first = f.path;
        g.firstIndex = index;
      }
    }
  }
  return [...groups.entries()]
    .map(([key, g]) => ({
      path: g.first,
      label: quantFromFilename(key) ?? (key.split('/').pop() ?? key),
      bytes: g.bytes,
      shards: g.shards,
    }))
    .sort((a, b) => a.bytes - b.bytes);
}

/** Q4_K_M when the repo has one (the most common local pick), else the smallest file. */
export function defaultGgufOption(options: GgufOption[]): GgufOption | undefined {
  return options.find((o) => o.label === 'Q4_K_M') ?? options[0];
}

/** Top-level `.safetensors` weight files summed (Mistral's duplicate `consolidated` file excluded). */
export function safetensorsBytes(files: RepoFile[]): number | undefined {
  const weights = files.filter((f) => !f.path.includes('/') && /\.safetensors$/i.test(f.path) && !/^consolidated/i.test(f.path));
  return weights.length ? weights.reduce((sum, f) => sum + f.size, 0) : undefined;
}

const GGUF_QUANT_KEYS: Record<string, WeightQuantKey> = {
  F32: 'fp32',
  BF16: 'bf16',
  F16: 'fp16',
  Q8_0: 'q8_0',
  Q6_K: 'q6_k',
  Q5_K_M: 'q5_k_m',
  Q4_K_M: 'q4_k_m',
  Q4_0: 'q4_0',
  IQ4_XS: 'iq4_xs',
  Q3_K_M: 'q3_k_m',
  Q2_K: 'q2_k',
};

// GGUF tag families -> nearest table quant, for tags not in the table when params are unknown.
const GGUF_FAMILIES: Array<[RegExp, WeightQuantKey]> = [
  [/^(IQ1|IQ2|Q2|TQ)/, 'q2_k'],
  [/^(IQ3|Q3)/, 'q3_k_m'],
  [/^IQ4/, 'iq4_xs'],
  [/^Q4_K/, 'q4_k_m'],
  [/^(Q4_0|Q4_1|MXFP4)/, 'q4_0'],
  [/^Q5/, 'q5_k_m'],
  [/^Q6/, 'q6_k'],
  [/^Q8/, 'q8_0'],
];

/** A table quant for some weight files; `exact` is false when it is only the closest one. */
export interface QuantMatch {
  quant: WeightQuantKey;
  exact: boolean;
}

/**
 * Table quant for a GGUF tag: the tag itself when it is in the table, else the nearest by
 * effective bits (params known), else by tag family (IQ2_M -> Q2_K). Undefined when nothing fits.
 */
export function ggufWeightQuant(label: string, bytes: number, params: number): QuantMatch | undefined {
  const direct = GGUF_QUANT_KEYS[label];
  if (direct) return { quant: direct, exact: true };
  const quant = closestWeightQuant(effectiveBitsPerWeight(bytes, params)) ?? GGUF_FAMILIES.find(([re]) => re.test(label))?.[1];
  return quant ? { quant, exact: false } : undefined;
}

/** Table quant for a pre-quantized safetensors repo; unknown methods get the nearest by effective bits. */
export function preQuantWeightQuant(q: PreQuant, bytes: number, params: number): QuantMatch | undefined {
  if (q.method === 'fp8') return { quant: 'fp8', exact: true };
  if ((q.method === 'awq' || q.method === 'gptq') && q.bits === 4) return { quant: 'awq_gptq_4bit', exact: true };
  if (q.method === 'bitsandbytes' && q.bits === 4) return { quant: 'nf4', exact: true };
  if (q.method === 'bitsandbytes' && q.bits === 8) return { quant: 'q8_0', exact: true };
  const quant = closestWeightQuant(effectiveBitsPerWeight(bytes, params));
  return quant ? { quant, exact: false } : undefined;
}

export type FetchRepoResult =
  | { ok: true; spec: ModelSpec; gguf?: { options: GgufOption[]; selected: string }; note?: string }
  | { ok: false; error: string };

const ggufError = (reason: string) =>
  `found GGUF files, but the header of the chosen file could not be read (${reason}); enter the architecture under Advanced, or look up the base repo`;

/** The note kept when weight files match no table quant: their bytes are not used. */
export const noQuantMatchNote = (label: string) =>
  `weight files found (${label}), but they match no weight quant in the table, so weights are estimated`;

/**
 * Attach file weights only with a table quant to tie them to; without one the bytes are
 * dropped and a note says so (file bytes never apply to a quant they are not in).
 */
function withFileWeights(spec: ModelSpec, bytes: number, label: string, match: QuantMatch | undefined): ModelSpec {
  if (!match) return { ...spec, warnings: [...spec.warnings, noQuantMatchNote(label)] };
  const fileWeights: FileWeights = {
    bytes,
    label: match.exact ? label : `${label} (closest table quant: ${WEIGHT_QUANTS[match.quant].label})`,
    quant: match.quant,
  };
  return { ...spec, fileWeights };
}

/** Read at most `limit` bytes of a response, even when the server ignored the Range header. */
async function readUpTo(res: Response, limit: number): Promise<ArrayBuffer> {
  if (res.status === 206 || !res.body) return (await res.arrayBuffer()).slice(0, limit);
  const reader = res.body.getReader();
  const out = new Uint8Array(limit);
  let n = 0;
  while (n < limit) {
    const { done, value } = await reader.read();
    if (done) break;
    const take = Math.min(value.byteLength, limit - n);
    out.set(value.subarray(0, take), n);
    n += take;
  }
  void reader.cancel().catch(() => undefined);
  return out.buffer.slice(0, n);
}

/**
 * fetchModel, plus the repo's file listing: exact weight bytes for pre-quantized safetensors
 * repos, and the whole GGUF path (file choice, ranged header read) for GGUF repos.
 * Never throws: every failure comes back as { ok: false, error }.
 */
export async function fetchRepo(
  rawId: string,
  token?: string,
  ggufPath?: string,
  fetchImpl: typeof fetch = fetch,
): Promise<FetchRepoResult> {
  const id = normalizeModelId(rawId ?? '');
  if (!id || !id.includes('/')) return { ok: false, error: HF_ERRORS.emptyId };
  const headers: Record<string, string> = {};
  if (token && token.trim()) headers.Authorization = `Bearer ${token.trim()}`;
  const path = id.split('/').map(encodeURIComponent).join('/');
  const listingUrl = `https://huggingface.co/api/models/${path}?blobs=true&expand[]=siblings&expand[]=gguf`;

  const listingPromise = (async (): Promise<RepoListing | undefined> => {
    try {
      const res = await fetchImpl(listingUrl, { headers });
      return res.ok ? parseRepoListing(await res.json()) : undefined;
    } catch {
      return undefined; // the listing is best-effort
    }
  })();
  const [model, listing] = await Promise.all([fetchModel(id, token, fetchImpl), listingPromise]);
  const files = listing?.files ?? [];

  const options = ggufOptions(files);
  const hasSafetensors = files.some((f) => /\.safetensors$/i.test(f.path));
  if (options.length > 0 && !hasSafetensors) {
    const option = options.find((o) => o.path === ggufPath) ?? defaultGgufOption(options);
    if (option) {
      const gguf = await fetchGguf(id, path, headers, option, options, listing?.ggufParams, fetchImpl);
      if (gguf.ok) return gguf;
      // The repo also has a usable config.json: keep today's behaviour, with a note.
      if (model.ok) {
        return { ok: true, spec: model.spec, note: `GGUF header not read (${gguf.reason}); architecture from config.json, weights estimated` };
      }
      return { ok: false, error: gguf.error };
    }
  }

  if (!model.ok || !model.quant) return model;
  const bytes = safetensorsBytes(files);
  if (bytes === undefined) return { ok: true, spec: model.spec };
  const match = preQuantWeightQuant(model.quant, bytes, model.spec.params);
  return {
    ok: true,
    spec: withFileWeights(model.spec, bytes, `${model.quant.method.toUpperCase()} safetensors`, match),
  };
}

async function fetchGguf(
  id: string,
  path: string,
  headers: Record<string, string>,
  option: GgufOption,
  options: GgufOption[],
  params: number | undefined,
  fetchImpl: typeof fetch,
): Promise<Extract<FetchRepoResult, { ok: true }> | { ok: false; error: string; reason: string }> {
  const fail = (reason: string, error = ggufError(reason)) => ({ ok: false as const, error, reason });
  const fileUrl = `https://huggingface.co/${path}/resolve/main/${option.path.split('/').map(encodeURIComponent).join('/')}`;
  let buf: ArrayBuffer;
  try {
    const res = await fetchImpl(fileUrl, { headers: { ...headers, Range: `bytes=0-${GGUF_HEADER_BYTES - 1}` } });
    if (res.status === 401 || res.status === 403) return fail(`HTTP ${res.status}`, HF_ERRORS.gated);
    if (!res.ok) return fail(`HTTP ${res.status}`);
    buf = await readUpTo(res, GGUF_HEADER_BYTES);
  } catch {
    return fail('network or CORS error');
  }
  let spec: ModelSpec;
  try {
    spec = parseGgufSpec(parseGgufHeader(buf), params, id);
  } catch (e) {
    return fail(e instanceof Error ? e.message : 'unreadable header');
  }
  const match = ggufWeightQuant(option.label, option.bytes, spec.params);
  return {
    ok: true,
    spec: withFileWeights(spec, option.bytes, `${option.label} GGUF`, match),
    gguf: { options, selected: option.path },
  };
}
