export type GpuVendor = 'nvidia-consumer' | 'nvidia-datacenter' | 'amd' | 'apple' | 'other';

export interface GpuPreset {
  name: string;
  vendor: GpuVendor;
  /** Per GPU, vendor-style decimal GB (unified memory for Apple / DGX Spark). */
  vramGB: number;
  /** Per GPU, GB/s. */
  bandwidthGBs: number;
}

export const CUSTOM_GPU_NAME = 'Custom';

export const GPU_PRESETS: GpuPreset[] = [
  // NVIDIA consumer / workstation
  { name: 'RTX 2070', vendor: 'nvidia-consumer', vramGB: 8, bandwidthGBs: 448 },
  { name: 'RTX 3060', vendor: 'nvidia-consumer', vramGB: 12, bandwidthGBs: 360 },
  { name: 'RTX 3090', vendor: 'nvidia-consumer', vramGB: 24, bandwidthGBs: 936 },
  { name: 'RTX 4070 Ti Super', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 672 },
  { name: 'RTX 4080', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 717 },
  { name: 'RTX 4090', vendor: 'nvidia-consumer', vramGB: 24, bandwidthGBs: 1008 },
  { name: 'RTX 5080', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 960 },
  { name: 'RTX 5090', vendor: 'nvidia-consumer', vramGB: 32, bandwidthGBs: 1792 },
  { name: 'RTX 6000 Ada', vendor: 'nvidia-consumer', vramGB: 48, bandwidthGBs: 960 },
  { name: 'RTX PRO 6000 Blackwell', vendor: 'nvidia-consumer', vramGB: 96, bandwidthGBs: 1792 },
  // NVIDIA datacenter
  { name: 'L4', vendor: 'nvidia-datacenter', vramGB: 24, bandwidthGBs: 300 },
  { name: 'L40S', vendor: 'nvidia-datacenter', vramGB: 48, bandwidthGBs: 864 },
  { name: 'A10', vendor: 'nvidia-datacenter', vramGB: 24, bandwidthGBs: 600 },
  { name: 'A100 40GB', vendor: 'nvidia-datacenter', vramGB: 40, bandwidthGBs: 1555 },
  { name: 'A100 80GB', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 2039 },
  { name: 'H100 PCIe', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 2000 },
  { name: 'H100 SXM', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 3350 },
  { name: 'H200', vendor: 'nvidia-datacenter', vramGB: 141, bandwidthGBs: 4800 },
  { name: 'B200', vendor: 'nvidia-datacenter', vramGB: 192, bandwidthGBs: 8000 },
  // AMD
  { name: 'AMD MI300X', vendor: 'amd', vramGB: 192, bandwidthGBs: 5300 },
  { name: 'Radeon RX 7900 XTX', vendor: 'amd', vramGB: 24, bandwidthGBs: 960 },
  // Apple (unified memory)
  { name: 'Apple M4 Max', vendor: 'apple', vramGB: 128, bandwidthGBs: 546 },
  { name: 'Apple M3 Ultra', vendor: 'apple', vramGB: 512, bandwidthGBs: 819 },
  { name: 'Apple M2 Ultra', vendor: 'apple', vramGB: 192, bandwidthGBs: 800 },
  // Other
  { name: 'NVIDIA DGX Spark', vendor: 'other', vramGB: 128, bandwidthGBs: 273 },
  { name: CUSTOM_GPU_NAME, vendor: 'other', vramGB: 24, bandwidthGBs: 1000 },
];

export function findGpuPreset(name: string): GpuPreset | undefined {
  return GPU_PRESETS.find((g) => g.name === name);
}
