export type GpuVendor = 'nvidia-consumer' | 'nvidia-datacenter' | 'amd' | 'apple' | 'other';

export interface GpuPreset {
  name: string;
  vendor: GpuVendor;
  /** Per GPU, vendor-style decimal GB (unified memory for Apple / DGX Spark). */
  vramGB: number;
  /** Per GPU, GB/s. */
  bandwidthGBs: number;
  /** Per GPU, dense (no sparsity) BF16 tensor TFLOPS; used for the prefill/TTFT estimate. */
  tflopsBf16: number;
}

export const CUSTOM_GPU_NAME = 'Custom';

// tflopsBf16: vendor spec-sheet dense (no-sparsity) FP16/BF16 tensor TFLOPS. NVIDIA and AMD
// datacenter parts publish this figure directly (sparsity figures are 2× and are not used here).
// Consumer cards' "AI TOPS" marketing figures are INT8-with-sparsity, so those are halved back
// to a dense FP16 estimate. Apple and DGX Spark do not publish a TFLOPS figure; those are rough
// estimates from third-party GPU-core benchmarks.
export const GPU_PRESETS: GpuPreset[] = [
  // NVIDIA consumer / workstation
  { name: 'RTX 2070', vendor: 'nvidia-consumer', vramGB: 8, bandwidthGBs: 448, tflopsBf16: 14.9 },
  { name: 'RTX 3060', vendor: 'nvidia-consumer', vramGB: 12, bandwidthGBs: 360, tflopsBf16: 12.7 },
  { name: 'RTX 3090', vendor: 'nvidia-consumer', vramGB: 24, bandwidthGBs: 936, tflopsBf16: 35.6 },
  { name: 'RTX 4070 Ti Super', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 672, tflopsBf16: 44.1 },
  { name: 'RTX 4080', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 717, tflopsBf16: 48.7 },
  { name: 'RTX 4090', vendor: 'nvidia-consumer', vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 82.6 },
  { name: 'RTX 5080', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 960, tflopsBf16: 56.3 },
  { name: 'RTX 5090', vendor: 'nvidia-consumer', vramGB: 32, bandwidthGBs: 1792, tflopsBf16: 104.8 },
  { name: 'RTX 6000 Ada', vendor: 'nvidia-consumer', vramGB: 48, bandwidthGBs: 960, tflopsBf16: 91.1 },
  { name: 'RTX PRO 6000 Blackwell', vendor: 'nvidia-consumer', vramGB: 96, bandwidthGBs: 1792, tflopsBf16: 126.0 },
  // NVIDIA datacenter
  { name: 'L4', vendor: 'nvidia-datacenter', vramGB: 24, bandwidthGBs: 300, tflopsBf16: 121.0 },
  { name: 'L40S', vendor: 'nvidia-datacenter', vramGB: 48, bandwidthGBs: 864, tflopsBf16: 181.0 },
  { name: 'A10', vendor: 'nvidia-datacenter', vramGB: 24, bandwidthGBs: 600, tflopsBf16: 62.5 },
  { name: 'A100 40GB', vendor: 'nvidia-datacenter', vramGB: 40, bandwidthGBs: 1555, tflopsBf16: 312.0 },
  { name: 'A100 80GB', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 2039, tflopsBf16: 312.0 },
  { name: 'H100 PCIe', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 2000, tflopsBf16: 756.5 },
  { name: 'H100 SXM', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5 },
  { name: 'H200', vendor: 'nvidia-datacenter', vramGB: 141, bandwidthGBs: 4800, tflopsBf16: 989.5 },
  { name: 'B200', vendor: 'nvidia-datacenter', vramGB: 192, bandwidthGBs: 8000, tflopsBf16: 1125.0 },
  // AMD
  { name: 'AMD MI300X', vendor: 'amd', vramGB: 192, bandwidthGBs: 5300, tflopsBf16: 1307.4 },
  { name: 'Radeon RX 7900 XTX', vendor: 'amd', vramGB: 24, bandwidthGBs: 960, tflopsBf16: 122.8 },
  // Apple (unified memory) — estimated from GPU core counts, not vendor-published
  { name: 'Apple M4 Max', vendor: 'apple', vramGB: 128, bandwidthGBs: 546, tflopsBf16: 34.0 },
  { name: 'Apple M3 Ultra', vendor: 'apple', vramGB: 512, bandwidthGBs: 819, tflopsBf16: 56.8 },
  { name: 'Apple M2 Ultra', vendor: 'apple', vramGB: 192, bandwidthGBs: 800, tflopsBf16: 54.4 },
  // Other
  { name: 'NVIDIA DGX Spark', vendor: 'other', vramGB: 128, bandwidthGBs: 273, tflopsBf16: 62.5 },
  { name: CUSTOM_GPU_NAME, vendor: 'other', vramGB: 24, bandwidthGBs: 1000, tflopsBf16: 100.0 },
];

export function findGpuPreset(name: string): GpuPreset | undefined {
  return GPU_PRESETS.find((g) => g.name === name);
}
