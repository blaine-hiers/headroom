import {
  activeParamsDetailed,
  CUSTOM_GPU_NAME,
  decodeState,
  defaultWeightQuantFor,
  findGpuPreset,
  findModelPreset,
  MODEL_PRESETS,
  refreshWarnings,
} from '../lib';
import type { CalcState, HardwareSpec, ModelSpec, Quant, Workload } from '../lib';

export const MIN_CONTEXT = 256;
export const MAX_USERS = 512;
export const MAX_GPUS = 16;
/** Upper bound for the integer shape fields (heads, dims, vocab, ...), matching the Advanced inputs. */
export const MAX_DIM = 1e7;
export const MAX_PARAMS = 1e14;

const DEFAULT_MODEL = findModelPreset('meta-llama/Llama-3.3-70B-Instruct') ?? MODEL_PRESETS[0];
const DEFAULT_GPU = findGpuPreset('RTX 4090');

export const defaultState: CalcState = {
  model: DEFAULT_MODEL,
  quant: { weight: defaultWeightQuantFor(DEFAULT_MODEL.nativeDtype), kv: 'fp16' },
  hardware: {
    gpuName: DEFAULT_GPU?.name ?? CUSTOM_GPU_NAME,
    gpuCount: 1,
    vramGB: DEFAULT_GPU?.vramGB ?? 24,
    bandwidthGBs: DEFAULT_GPU?.bandwidthGBs ?? 1008,
    reservePct: 5,
    overheadGB: 1,
  },
  workload: { contextTokens: 8192, concurrentUsers: 1 },
};

export function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

/** Upper bound of the context control for a model (never below the minimum). */
export function maxContextFor(model: ModelSpec): number {
  return Math.max(MIN_CONTEXT, Math.floor(model.maxPositionEmbeddings) || MIN_CONTEXT);
}

export type Action =
  | { type: 'loadModel'; spec: ModelSpec }
  | { type: 'editModel'; patch: Partial<ModelSpec> }
  | { type: 'quant'; patch: Partial<Quant> }
  | { type: 'hardware'; patch: Partial<HardwareSpec> }
  | { type: 'workload'; patch: Partial<Workload> };

function clampWorkload(w: Workload, model: ModelSpec): Workload {
  return {
    contextTokens: Math.round(clamp(w.contextTokens, MIN_CONTEXT, maxContextFor(model))),
    concurrentUsers: Math.round(clamp(w.concurrentUsers, 1, MAX_USERS)),
  };
}

export function reducer(state: CalcState, action: Action): CalcState {
  switch (action.type) {
    case 'loadModel': {
      const model = action.spec;
      return {
        ...state,
        model,
        quant: { ...state.quant, weight: defaultWeightQuantFor(model.nativeDtype) },
        workload: clampWorkload(state.workload, model),
      };
    }
    case 'editModel': {
      const merged: ModelSpec = { ...state.model, ...action.patch, source: 'manual' };
      // `moe: undefined` in a patch means "dense": drop the key rather than keep an undefined.
      if ('moe' in action.patch && action.patch.moe === undefined) delete merged.moe;
      merged.activeParams = activeParamsDetailed(merged).active;
      merged.warnings = refreshWarnings(merged);
      return { ...state, model: merged, workload: clampWorkload(state.workload, merged) };
    }
    case 'quant':
      return { ...state, quant: { ...state.quant, ...action.patch } };
    case 'hardware': {
      const hw = { ...state.hardware, ...action.patch };
      const p = action.patch;
      if (p.gpuName !== undefined && p.gpuName !== CUSTOM_GPU_NAME) {
        const preset = findGpuPreset(p.gpuName);
        if (preset) {
          hw.vramGB = preset.vramGB;
          hw.bandwidthGBs = preset.bandwidthGBs;
        }
      } else if (p.gpuName === undefined && (p.vramGB !== undefined || p.bandwidthGBs !== undefined)) {
        hw.gpuName = CUSTOM_GPU_NAME;
      }
      return { ...state, hardware: hw };
    }
    case 'workload':
      return { ...state, workload: clampWorkload({ ...state.workload, ...action.patch }, state.model) };
  }
}

const int = (v: number, lo: number, hi: number) => Math.round(clamp(v, lo, hi));
const optInt = (v: number | undefined, lo: number, hi: number) => (v === undefined ? undefined : int(v, lo, hi));

/** Pull every decoded model field into the range the Advanced inputs allow. */
export function clampModel(m: ModelSpec): ModelSpec {
  const numLayers = int(m.numLayers, 1, 1000);
  const out: ModelSpec = {
    ...m,
    params: Math.round(clamp(m.params, 0, MAX_PARAMS)),
    numLayers,
    numKvHeads: int(m.numKvHeads, 0, MAX_DIM),
    headDim: int(m.headDim, 0, MAX_DIM),
    maxPositionEmbeddings: int(m.maxPositionEmbeddings, MIN_CONTEXT, MAX_DIM * 10),
    hiddenSize: int(m.hiddenSize, 0, MAX_DIM),
    vocabSize: int(m.vocabSize, 0, MAX_DIM),
  };
  const opt = {
    kvLoraRank: optInt(m.kvLoraRank, 0, MAX_DIM),
    qkRopeHeadDim: optInt(m.qkRopeHeadDim, 0, MAX_DIM),
    slidingWindow: optInt(m.slidingWindow, 0, MAX_DIM),
    slidingLayers: optInt(m.slidingLayers, 0, numLayers),
  };
  for (const [k, v] of Object.entries(opt) as Array<[keyof typeof opt, number | undefined]>) {
    if (v === undefined) delete out[k];
    else out[k] = v;
  }
  if (m.moe) {
    out.moe = {
      numExperts: int(m.moe.numExperts, 1, 100000),
      expertsPerToken: int(m.moe.expertsPerToken, 1, 100000),
      sharedExperts: int(m.moe.sharedExperts, 0, 100000),
    };
  }
  if (m.ffn) {
    const f = m.ffn;
    out.ffn = { intermediateSize: int(f.intermediateSize, 0, MAX_DIM), numAttentionHeads: int(f.numAttentionHeads, 0, MAX_DIM), tieEmbeddings: f.tieEmbeddings };
    const moeInt = optInt(f.moeIntermediateSize, 0, MAX_DIM);
    const firstK = optInt(f.firstKDense, 0, numLayers);
    const qLora = optInt(f.qLoraRank, 0, MAX_DIM);
    const vHead = optInt(f.vHeadDim, 0, MAX_DIM);
    if (moeInt !== undefined) out.ffn.moeIntermediateSize = moeInt;
    if (firstK !== undefined) out.ffn.firstKDense = firstK;
    if (qLora !== undefined) out.ffn.qLoraRank = qLora;
    if (vHead !== undefined) out.ffn.vHeadDim = vHead;
  }
  out.activeParams = activeParamsDetailed(out).active;
  return out;
}

/** Initial state from the URL; anything unparseable falls back to the defaults. */
export function initialState(search: string): CalcState {
  const s = decodeState(search, defaultState);
  const hw = s.hardware;
  const model = clampModel(s.model);
  model.warnings = refreshWarnings(model);
  return {
    ...s,
    model,
    hardware: {
      ...hw,
      // An unknown GPU name keeps the link's VRAM and bandwidth under the Custom entry.
      gpuName: findGpuPreset(hw.gpuName) ? hw.gpuName : CUSTOM_GPU_NAME,
      gpuCount: Math.round(clamp(hw.gpuCount, 1, MAX_GPUS)),
      vramGB: clamp(hw.vramGB, 0.1, 4096),
      bandwidthGBs: clamp(hw.bandwidthGBs, 1, 100000),
      reservePct: clamp(hw.reservePct, 0, 50),
      overheadGB: clamp(hw.overheadGB, 0, 8),
    },
    workload: clampWorkload(s.workload, model),
  };
}

export type FitLevel = 'fits' | 'tight' | 'nofit';

/** Tight = fits with under 10% of usable VRAM left over. */
export function fitLevel(fits: boolean, headroomBytes: number, usableBytes: number): FitLevel {
  if (!fits) return 'nofit';
  if (usableBytes > 0 && headroomBytes / usableBytes < 0.1) return 'tight';
  return 'fits';
}
