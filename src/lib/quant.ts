import type { KvQuantKey, NativeDtype, WeightQuantKey } from './types';

export interface WeightQuantInfo {
  label: string;
  bitsPerWeight: number;
  note?: string;
}

export interface KvQuantInfo {
  label: string;
  bytesPerElement: number;
}

/** Effective bits per weight (approximate, llama.cpp effective sizes incl. scales). */
export const WEIGHT_QUANTS: Record<WeightQuantKey, WeightQuantInfo> = {
  fp32: { label: 'FP32', bitsPerWeight: 32 },
  bf16: { label: 'BF16', bitsPerWeight: 16, note: 'native for most models' },
  fp16: { label: 'FP16', bitsPerWeight: 16 },
  fp8: { label: 'FP8', bitsPerWeight: 8, note: 'native DeepSeek V3, vLLM FP8' },
  q8_0: { label: 'Q8_0 / INT8', bitsPerWeight: 8.5 },
  q6_k: { label: 'Q6_K', bitsPerWeight: 6.56 },
  q5_k_m: { label: 'Q5_K_M', bitsPerWeight: 5.69 },
  q4_k_m: { label: 'Q4_K_M', bitsPerWeight: 4.85, note: 'most common GGUF' },
  q4_0: { label: 'Q4_0', bitsPerWeight: 4.5 },
  awq_gptq_4bit: { label: 'AWQ / GPTQ 4-bit (g128)', bitsPerWeight: 4.5 },
  iq4_xs: { label: 'IQ4_XS', bitsPerWeight: 4.25 },
  nf4: { label: 'NF4 (bitsandbytes)', bitsPerWeight: 4.5 },
  q3_k_m: { label: 'Q3_K_M', bitsPerWeight: 3.91 },
  q2_k: { label: 'Q2_K', bitsPerWeight: 3.35 },
};

export const KV_QUANTS: Record<KvQuantKey, KvQuantInfo> = {
  fp16: { label: 'FP16', bytesPerElement: 2 },
  bf16: { label: 'BF16', bytesPerElement: 2 },
  fp8: { label: 'FP8', bytesPerElement: 1 },
  int8: { label: 'INT8', bytesPerElement: 1 },
  int4: { label: 'INT4', bytesPerElement: 0.5 },
};

export function defaultWeightQuantFor(nativeDtype: NativeDtype): WeightQuantKey {
  switch (nativeDtype) {
    case 'fp32':
      return 'fp32';
    case 'fp16':
      return 'fp16';
    case 'fp8':
      return 'fp8';
    default:
      return 'bf16';
  }
}
