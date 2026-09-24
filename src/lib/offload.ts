import { DECODE_EFFICIENCY } from './throughput';
import type { ThroughputResult } from './throughput';
import type { OffloadPlan, OffloadSpec } from './types';

/** Offload is off by default: today's fit/throughput numbers stay unchanged until a user opts in. */
export const DEFAULT_OFFLOAD: OffloadSpec = { enabled: false, systemRamGB: 64, ramBandwidthGBs: 50 };

/** `hardware.offload` is optional (old shared links never had it); this fills in the disabled default. */
export function resolveOffload(offload: OffloadSpec | undefined): OffloadSpec {
  return offload ?? DEFAULT_OFFLOAD;
}

export interface RamPreset {
  label: string;
  /** GB/s. Apple is n/a here — its unified memory bandwidth is the GPU preset's own bandwidthGBs. */
  bandwidthGBs: number | null;
}

/** A small bundled list for the RAM-bandwidth field; DDR4/DDR5 figures are dual-channel ballparks. */
export const RAM_PRESETS: RamPreset[] = [
  { label: 'DDR4 dual-channel (~50 GB/s)', bandwidthGBs: 50 },
  { label: 'DDR5 dual-channel (~80-100 GB/s)', bandwidthGBs: 90 },
  { label: 'Apple unified memory (use the GPU bandwidth above)', bandwidthGBs: null },
];

export interface PlanOffloadInput {
  /** Total weight bytes for the whole model (all layers, embeddings and LM head). */
  weightBytes: number;
  numLayers: number;
  /** GPU bytes left for weights after KV and runtime overhead are already accounted for. May be negative. */
  usableGpuBytes: number;
  offload: OffloadSpec;
}

/**
 * Splits the model's layers between GPU and system RAM, llama.cpp's `-ngl` split.
 * `bytesPerLayer ≈ weightBytes / numLayers` spreads the embeddings/LM head evenly across the
 * layer count rather than placing them structurally — an approximation, like the rest of this
 * calculator, and "sensible" here just means the result is always clamped to [0, numLayers].
 * When offload is disabled, every layer is treated as GPU-resident (today's behavior).
 */
export function planOffload({ weightBytes, numLayers, usableGpuBytes, offload }: PlanOffloadInput): OffloadPlan {
  const layers = Math.max(0, Math.floor(numLayers));
  const bytesPerLayer = layers > 0 ? weightBytes / layers : 0;

  if (!offload.enabled) {
    return { gpuLayers: layers, cpuLayers: 0, bytesPerLayer, gpuWeightBytes: weightBytes, cpuWeightBytes: 0, fitsInRam: true };
  }

  const rawGpuLayers = bytesPerLayer > 0 ? Math.floor(usableGpuBytes / bytesPerLayer) : layers;
  const gpuLayers = Math.min(layers, Math.max(0, rawGpuLayers));
  const cpuLayers = layers - gpuLayers;
  const gpuWeightBytes = gpuLayers * bytesPerLayer;
  const cpuWeightBytes = weightBytes - gpuWeightBytes;
  const fitsInRam = cpuWeightBytes <= offload.systemRamGB * 1e9;

  return { gpuLayers, cpuLayers, bytesPerLayer, gpuWeightBytes, cpuWeightBytes, fitsInRam };
}

/** Splits `bytes` between GPU and CPU in proportion to the GPU/total layer counts. */
export function splitByLayerFraction(bytes: number, gpuLayers: number, numLayers: number): { gpu: number; cpu: number } {
  const layers = Math.max(0, Math.floor(numLayers));
  if (layers <= 0) return { gpu: bytes, cpu: 0 };
  const gpu = bytes * (Math.max(0, gpuLayers) / layers);
  return { gpu, cpu: bytes - gpu };
}

export interface OffloadThroughputInput {
  /** Active-weight bytes read per decode step for the GPU-resident layers. */
  gpuActiveWeightBytes: number;
  /** Active-weight bytes read per decode step for the RAM-resident layers. */
  cpuActiveWeightBytes: number;
  /** KV bytes per request — KV stays on the GPU (llama.cpp's default), so it never counts against RAM bandwidth. */
  kvBytesPerRequest: number;
  concurrentUsers: number;
  /** Per GPU, GB/s. */
  bandwidthGBs: number;
  gpuCount: number;
  /** System RAM bandwidth, GB/s. */
  ramBandwidthGBs: number;
  /** Same efficiency the GPU-only throughput estimate uses (DECODE_EFFICIENCY × tensor-parallel penalty). */
  gpuEfficiency: number;
}

/**
 * Bandwidth-bound decode estimate when layers are split across GPU and system RAM:
 * time per step = GPU bytes / GPU bandwidth + CPU bytes / RAM bandwidth, each already scaled
 * by an efficiency factor. The RAM term uses the plain decode efficiency (no tensor-parallel
 * penalty — that penalty models GPU-to-GPU communication, which doesn't apply to system RAM).
 */
export function offloadDecodeThroughput(input: OffloadThroughputInput): ThroughputResult {
  const users = Math.max(0, Math.floor(input.concurrentUsers));
  const gpuBandwidth = input.bandwidthGBs * 1e9 * input.gpuCount * input.gpuEfficiency;
  const ramBandwidth = input.ramBandwidthGBs * 1e9 * DECODE_EFFICIENCY;
  const gpuBytes = input.gpuActiveWeightBytes + users * input.kvBytesPerRequest;
  const cpuBytes = Math.max(0, input.cpuActiveWeightBytes);

  const gpuTime = gpuBandwidth > 0 ? gpuBytes / gpuBandwidth : gpuBytes > 0 ? Infinity : 0;
  const cpuTime = ramBandwidth > 0 ? cpuBytes / ramBandwidth : cpuBytes > 0 ? Infinity : 0;
  const timePerStep = gpuTime + cpuTime;
  const stepsPerSec = timePerStep > 0 && Number.isFinite(timePerStep) ? 1 / timePerStep : 0;

  return { perUserTokS: stepsPerSec, aggregateTokS: stepsPerSec * users, efficiency: input.gpuEfficiency };
}
