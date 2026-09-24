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

/** Vendor groups in display order, shared by the GPU select and the hardware finder's filter. */
export const GPU_VENDOR_GROUPS: Array<{ vendor: GpuVendor; label: string }> = [
  { vendor: 'nvidia-consumer', label: 'NVIDIA consumer / workstation' },
  { vendor: 'nvidia-datacenter', label: 'NVIDIA datacenter' },
  { vendor: 'amd', label: 'AMD' },
  { vendor: 'apple', label: 'Apple (unified memory)' },
  { vendor: 'other', label: 'Other' },
];

// tflopsBf16: dense (no 2:4-sparsity) FP16/BF16 tensor-core TFLOPS with FP32 accumulate.
// NVIDIA datacenter parts (A10, A100, H100, H200, B200, L4, L40S) and AMD MI300X publish this
// figure directly in their datasheets/architecture whitepapers. For GeForce/RTX-workstation
// (Ampere/Ada/Blackwell) parts, NVIDIA's own consumer spec sheets quote the FP16-with-FP16-
// accumulate, 2:4-sparse figure (marketed as "AI TOPS"); the dense/FP32-accumulate rate used
// here is that figure ÷4 (÷2 for sparsity, ÷2 for the accumulate precision), taken from or
// cross-checked against the RTX Blackwell / Ada Lovelace / Ampere architecture whitepapers.
// Apple and DGX Spark do not publish a dense tensor TFLOPS figure at all; those two are
// estimates (GPU core count × clock for Apple; NVIDIA's published "1 PFLOP FP4 2:4-sparse"
// DGX Spark figure, halved twice for FP8 then FP16 dense).
export const GPU_PRESETS: GpuPreset[] = [
  // NVIDIA consumer / workstation
  { name: 'RTX 2070', vendor: 'nvidia-consumer', vramGB: 8, bandwidthGBs: 448, tflopsBf16: 14.9 }, // Turing whitepaper, FP16 dense w/ FP32 accumulate
  { name: 'RTX 3060', vendor: 'nvidia-consumer', vramGB: 12, bandwidthGBs: 360, tflopsBf16: 25.4 }, // Ampere GA106: scaled from RTX 3090 by SM count × clock
  { name: 'RTX 3090', vendor: 'nvidia-consumer', vramGB: 24, bandwidthGBs: 936, tflopsBf16: 71.0 }, // Ampere GA102 architecture whitepaper
  { name: 'RTX 4070 Ti Super', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 672, tflopsBf16: 88.0 }, // Ada AD103 datasheet
  { name: 'RTX 4080', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 717, tflopsBf16: 97.5 }, // Ada AD103 datasheet
  { name: 'RTX 4090', vendor: 'nvidia-consumer', vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 165.0 }, // Ada AD102 architecture whitepaper
  { name: 'RTX 5080', vendor: 'nvidia-consumer', vramGB: 16, bandwidthGBs: 960, tflopsBf16: 112.6 }, // Blackwell GB203: scaled from RTX 5090 by SM count × clock
  { name: 'RTX 5090', vendor: 'nvidia-consumer', vramGB: 32, bandwidthGBs: 1792, tflopsBf16: 209.5 }, // RTX Blackwell GPU architecture whitepaper
  { name: 'RTX 6000 Ada', vendor: 'nvidia-consumer', vramGB: 48, bandwidthGBs: 960, tflopsBf16: 364.0 }, // RTX 6000 Ada datasheet: 1457 TFLOPS FP8 with sparsity, halved for dense and again for FP16. Workstation cards run FP32 accumulate at full rate, unlike GeForce (cf. L40S, same AD102)
  { name: 'RTX PRO 6000 Blackwell', vendor: 'nvidia-consumer', vramGB: 96, bandwidthGBs: 1792, tflopsBf16: 500.0 }, // estimate: datasheet 4000 AI TOPS (FP4 with sparsity) halved for dense, FP8 and FP16. Workstation cards run FP32 accumulate at full rate, unlike GeForce
  // NVIDIA datacenter
  { name: 'L4', vendor: 'nvidia-datacenter', vramGB: 24, bandwidthGBs: 300, tflopsBf16: 121.0 }, // NVIDIA L4 datasheet (FP16 Tensor Core, dense)
  { name: 'L40S', vendor: 'nvidia-datacenter', vramGB: 48, bandwidthGBs: 864, tflopsBf16: 362.0 }, // NVIDIA L40S datasheet (FP16 Tensor Core, dense)
  { name: 'A10', vendor: 'nvidia-datacenter', vramGB: 24, bandwidthGBs: 600, tflopsBf16: 125.0 }, // NVIDIA A10 datasheet (FP16 Tensor Core, dense)
  { name: 'A100 40GB', vendor: 'nvidia-datacenter', vramGB: 40, bandwidthGBs: 1555, tflopsBf16: 312.0 }, // NVIDIA A100 datasheet (FP16/BF16 Tensor Core, dense)
  { name: 'A100 80GB', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 2039, tflopsBf16: 312.0 }, // NVIDIA A100 datasheet (FP16/BF16 Tensor Core, dense)
  { name: 'H100 PCIe', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 2000, tflopsBf16: 756.5 }, // NVIDIA H100 datasheet (FP16/BF16 Tensor Core, dense)
  { name: 'H100 SXM', vendor: 'nvidia-datacenter', vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5 }, // NVIDIA H100 datasheet (FP16/BF16 Tensor Core, dense)
  { name: 'H200', vendor: 'nvidia-datacenter', vramGB: 141, bandwidthGBs: 4800, tflopsBf16: 989.5 }, // Same Hopper compute die as H100 SXM
  { name: 'B200', vendor: 'nvidia-datacenter', vramGB: 192, bandwidthGBs: 8000, tflopsBf16: 2250.0 }, // NVIDIA Blackwell architecture whitepaper (FP16/BF16 Tensor Core, dense)
  // AMD
  { name: 'AMD MI300X', vendor: 'amd', vramGB: 192, bandwidthGBs: 5300, tflopsBf16: 1307.4 }, // AMD MI300X datasheet (FP16/BF16 Matrix, dense)
  { name: 'Radeon RX 7900 XTX', vendor: 'amd', vramGB: 24, bandwidthGBs: 960, tflopsBf16: 122.8 }, // RDNA 3 datasheet (FP16 w/ FP16 accumulate)
  // Apple (unified memory) — estimate: GPU core count × ALU count × clock, doubled for FP16
  { name: 'Apple M4 Max', vendor: 'apple', vramGB: 128, bandwidthGBs: 546, tflopsBf16: 34.0 }, // estimate
  { name: 'Apple M3 Ultra', vendor: 'apple', vramGB: 512, bandwidthGBs: 819, tflopsBf16: 58.0 }, // estimate
  { name: 'Apple M2 Ultra', vendor: 'apple', vramGB: 192, bandwidthGBs: 800, tflopsBf16: 54.4 }, // estimate
  // Other
  { name: 'NVIDIA DGX Spark', vendor: 'other', vramGB: 128, bandwidthGBs: 273, tflopsBf16: 125.0 }, // estimate: NVIDIA's published "1 PFLOP FP4 2:4-sparse" figure, halved twice (FP8, then FP16 dense)
  { name: CUSTOM_GPU_NAME, vendor: 'other', vramGB: 24, bandwidthGBs: 1000, tflopsBf16: 100.0 },
];

export function findGpuPreset(name: string): GpuPreset | undefined {
  return GPU_PRESETS.find((g) => g.name === name);
}
