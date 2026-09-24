import { KV_QUANTS, WEIGHT_QUANTS } from './quant';
import type {
  Attention,
  CalcState,
  FfnSpec,
  KvQuantKey,
  ModelSpec,
  NativeDtype,
  WeightQuantKey,
} from './types';

// Compact query-string keys. Changing one breaks existing shared links.
const K = {
  id: 'id',
  name: 'n',
  params: 'p',
  activeParams: 'ap',
  numLayers: 'l',
  attention: 'at',
  numKvHeads: 'kh',
  headDim: 'hd',
  kvLoraRank: 'kr',
  qkRopeHeadDim: 'qr',
  slidingWindow: 'sw',
  slidingLayers: 'sl',
  maxPositionEmbeddings: 'mp',
  hiddenSize: 'hs',
  vocabSize: 'vs',
  nativeDtype: 'dt',
  moe: 'moe',
  ffn: 'ff',
  source: 'src',
  warnings: 'w',
  weightQuant: 'wq',
  kvQuant: 'kq',
  gpuName: 'g',
  gpuCount: 'gc',
  vramGB: 'vr',
  bandwidthGBs: 'bw',
  reservePct: 'rp',
  overheadGB: 'oh',
  appleWiredLimitGB: 'awl',
  contextTokens: 'c',
  concurrentUsers: 'u',
  fileWeightBytes: 'fwb',
  fileWeightLabel: 'fwl',
  fileWeightQuant: 'fwq',
} as const;

const ATTENTIONS: readonly Attention[] = ['mha_gqa', 'mla'];
const DTYPES: readonly NativeDtype[] = ['bf16', 'fp16', 'fp32', 'fp8'];
const SOURCES: readonly ModelSpec['source'][] = ['hf', 'preset', 'manual'];

export function encodeState(state: CalcState): string {
  const q = new URLSearchParams();
  const m = state.model;
  const set = (k: string, v: string | number | undefined) => {
    if (v !== undefined) q.set(k, String(v));
  };
  set(K.id, m.id);
  set(K.name, m.name);
  set(K.params, m.params);
  set(K.activeParams, m.activeParams);
  set(K.numLayers, m.numLayers);
  set(K.attention, m.attention);
  set(K.numKvHeads, m.numKvHeads);
  set(K.headDim, m.headDim);
  set(K.kvLoraRank, m.kvLoraRank);
  set(K.qkRopeHeadDim, m.qkRopeHeadDim);
  set(K.slidingWindow, m.slidingWindow);
  set(K.slidingLayers, m.slidingLayers);
  set(K.maxPositionEmbeddings, m.maxPositionEmbeddings);
  set(K.hiddenSize, m.hiddenSize);
  set(K.vocabSize, m.vocabSize);
  set(K.nativeDtype, m.nativeDtype);
  if (m.moe) set(K.moe, `${m.moe.numExperts},${m.moe.expertsPerToken},${m.moe.sharedExperts}`);
  if (m.ffn) set(K.ffn, encodeFfn(m.ffn));
  if (m.fileWeights) {
    set(K.fileWeightBytes, m.fileWeights.bytes);
    set(K.fileWeightLabel, m.fileWeights.label);
    set(K.fileWeightQuant, m.fileWeights.quant);
  }
  set(K.source, m.source);
  for (const w of m.warnings) q.append(K.warnings, w);
  set(K.weightQuant, state.quant.weight);
  set(K.kvQuant, state.quant.kv);
  const h = state.hardware;
  set(K.gpuName, h.gpuName);
  set(K.gpuCount, h.gpuCount);
  set(K.vramGB, h.vramGB);
  set(K.bandwidthGBs, h.bandwidthGBs);
  set(K.reservePct, h.reservePct);
  set(K.overheadGB, h.overheadGB);
  set(K.appleWiredLimitGB, h.appleWiredLimitGB);
  set(K.contextTokens, state.workload.contextTokens);
  set(K.concurrentUsers, state.workload.concurrentUsers);
  return q.toString();
}

class BadState extends Error {}

// ff=intermediate,heads,tie(0|1),moeIntermediate,firstKDense,qLoraRank,vHeadDim (optional ones may be empty)
function encodeFfn(f: FfnSpec): string {
  const o = (v: number | undefined) => (v === undefined ? '' : String(v));
  return [f.intermediateSize, f.numAttentionHeads, f.tieEmbeddings ? 1 : 0, o(f.moeIntermediateSize), o(f.firstKDense), o(f.qLoraRank), o(f.vHeadDim)].join(',');
}

function decodeFfn(raw: string): FfnSpec {
  const parts = raw.split(',').map((s) => s.trim());
  if (parts.length !== 7) throw new BadState();
  const req = (s: string): number => {
    const n = s === '' ? NaN : Number(s);
    if (!Number.isFinite(n)) throw new BadState();
    return n;
  };
  const opt = (s: string): number | undefined => (s === '' ? undefined : req(s));
  if (parts[2] !== '0' && parts[2] !== '1') throw new BadState();
  const f: FfnSpec = { intermediateSize: req(parts[0]), numAttentionHeads: req(parts[1]), tieEmbeddings: parts[2] === '1' };
  const moeIntermediateSize = opt(parts[3]);
  const firstKDense = opt(parts[4]);
  const qLoraRank = opt(parts[5]);
  const vHeadDim = opt(parts[6]);
  if (moeIntermediateSize !== undefined) f.moeIntermediateSize = moeIntermediateSize;
  if (firstKDense !== undefined) f.firstKDense = firstKDense;
  if (qLoraRank !== undefined) f.qLoraRank = qLoraRank;
  if (vHeadDim !== undefined) f.vHeadDim = vHeadDim;
  return f;
}

function oneOf<T extends string>(v: string | null, allowed: readonly T[]): T {
  if (v === null || !(allowed as readonly string[]).includes(v)) throw new BadState();
  return v as T;
}

/**
 * Decode a query string (with or without the leading "?"). Any missing or invalid
 * required field returns `fallback` unchanged; partial states are never produced.
 */
export function decodeState(qs: string, fallback: CalcState): CalcState {
  try {
    const q = new URLSearchParams(qs.startsWith('?') ? qs.slice(1) : qs);
    if (!q.has(K.id)) return fallback;

    const str = (k: string): string => {
      const v = q.get(k);
      if (v === null) throw new BadState();
      return v;
    };
    const num = (k: string): number => {
      const raw = q.get(k);
      if (raw === null || raw.trim() === '') throw new BadState();
      const n = Number(raw);
      if (!Number.isFinite(n)) throw new BadState();
      return n;
    };
    const optNum = (k: string): number | undefined => (q.has(k) ? num(k) : undefined);

    const model: ModelSpec = {
      id: str(K.id),
      name: str(K.name),
      params: num(K.params),
      activeParams: num(K.activeParams),
      numLayers: num(K.numLayers),
      attention: oneOf(q.get(K.attention), ATTENTIONS),
      numKvHeads: num(K.numKvHeads),
      headDim: num(K.headDim),
      maxPositionEmbeddings: num(K.maxPositionEmbeddings),
      hiddenSize: num(K.hiddenSize),
      vocabSize: num(K.vocabSize),
      nativeDtype: oneOf(q.get(K.nativeDtype), DTYPES),
      source: oneOf(q.get(K.source), SOURCES),
      warnings: q.getAll(K.warnings),
    };
    const kvLoraRank = optNum(K.kvLoraRank);
    const qkRopeHeadDim = optNum(K.qkRopeHeadDim);
    const slidingWindow = optNum(K.slidingWindow);
    const slidingLayers = optNum(K.slidingLayers);
    if (kvLoraRank !== undefined) model.kvLoraRank = kvLoraRank;
    if (qkRopeHeadDim !== undefined) model.qkRopeHeadDim = qkRopeHeadDim;
    if (slidingWindow !== undefined) model.slidingWindow = slidingWindow;
    if (slidingLayers !== undefined) model.slidingLayers = slidingLayers;
    const moeRaw = q.get(K.moe);
    if (moeRaw !== null) {
      const parts = moeRaw.split(',').map((s) => (s.trim() === '' ? NaN : Number(s)));
      if (parts.length !== 3 || !parts.every(Number.isFinite)) throw new BadState();
      model.moe = { numExperts: parts[0], expertsPerToken: parts[1], sharedExperts: parts[2] };
    }
    const ffnRaw = q.get(K.ffn);
    if (ffnRaw !== null) model.ffn = decodeFfn(ffnRaw);
    // File bytes only mean something with the quant they are in: without `fwq` they are ignored.
    if (q.has(K.fileWeightBytes) && q.has(K.fileWeightQuant)) {
      const bytes = num(K.fileWeightBytes);
      if (!(bytes > 0)) throw new BadState();
      const quant = oneOf(q.get(K.fileWeightQuant), Object.keys(WEIGHT_QUANTS) as WeightQuantKey[]);
      model.fileWeights = { bytes, label: q.get(K.fileWeightLabel) ?? '', quant };
    }

    return {
      model,
      quant: {
        weight: oneOf(q.get(K.weightQuant), Object.keys(WEIGHT_QUANTS) as WeightQuantKey[]),
        kv: oneOf(q.get(K.kvQuant), Object.keys(KV_QUANTS) as KvQuantKey[]),
      },
      hardware: (() => {
        const hw = {
          gpuName: str(K.gpuName),
          gpuCount: num(K.gpuCount),
          vramGB: num(K.vramGB),
          bandwidthGBs: num(K.bandwidthGBs),
          reservePct: num(K.reservePct),
          overheadGB: num(K.overheadGB),
        };
        const appleWiredLimit = optNum(K.appleWiredLimitGB);
        if (appleWiredLimit !== undefined) (hw as any).appleWiredLimitGB = appleWiredLimit;
        return hw;
      })(),
      workload: {
        contextTokens: num(K.contextTokens),
        concurrentUsers: num(K.concurrentUsers),
      },
    };
  } catch {
    return fallback;
  }
}
