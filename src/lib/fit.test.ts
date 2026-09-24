import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { calculate, effectiveVramGB, maxContext, maxUsers, overheadBytes, overheadBytesForRuntime, usableBytes, usableBytesForRuntime } from './fit';
import { findGpuPreset } from './presets/gpus';
import { findModelPreset } from './presets/models';
import { SGLANG_MEM_FRACTION_STATIC, VLLM_GPU_MEMORY_UTILIZATION, VLLM_KV_BLOCK_TOKENS, VLLM_OVERHEAD_ALLOWANCE_GB } from './runtime';
import { tensorParallelEfficiency } from './tensorParallel';
import type { CalcState, HardwareSpec, ModelSpec } from './types';

const h100x4: HardwareSpec = { gpuName: 'H100 SXM', gpuCount: 4, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5, reservePct: 5, overheadGB: 1 };

function state(overrides: Partial<CalcState> = {}): CalcState {
  return {
    model: makeSpec(),
    quant: { weight: 'bf16', kv: 'fp16' },
    hardware: h100x4,
    workload: { contextTokens: 8192, concurrentUsers: 4 },
    runtime: 'generic',
    ...overrides,
  };
}

describe('usable and overhead', () => {
  it('usable = count × GB × 1e9 × (1 − reserve)', () => {
    expect(usableBytes(h100x4)).toBeCloseTo(4 * 80e9 * 0.95, 0);
    expect(usableBytes({ ...h100x4, reservePct: 0 })).toBe(320e9);
  });
  it('overhead = GB × 1e9 × count', () => {
    expect(overheadBytes(h100x4)).toBe(4e9);
  });
});

describe('effectiveVramGB for Apple GPUs', () => {
  it('non-Apple GPUs return vramGB unchanged', () => {
    expect(effectiveVramGB('H100 SXM', 80)).toBe(80);
    expect(effectiveVramGB('RTX 4090', 24)).toBe(24);
    expect(effectiveVramGB('AMD MI300X', 192)).toBe(192);
  });

  it('Apple GPUs <= 36 GB apply 0.67x wired limit', () => {
    expect(effectiveVramGB('Apple M4 Max', 32, undefined)).toBeCloseTo(32 * 0.67, 1);
    expect(effectiveVramGB('Apple M4 Max', 36, undefined)).toBeCloseTo(36 * 0.67, 1);
  });

  it('Apple GPUs > 36 GB apply 0.75x wired limit', () => {
    expect(effectiveVramGB('Apple M2 Ultra', 192, undefined)).toBeCloseTo(192 * 0.75, 1);
    expect(effectiveVramGB('Apple M3 Ultra', 512, undefined)).toBeCloseTo(512 * 0.75, 1);
  });

  it('Apple GPU with override uses the custom limit', () => {
    expect(effectiveVramGB('Apple M2 Ultra', 192, 120)).toBe(120);
    expect(effectiveVramGB('Apple M4 Max', 32, 28)).toBe(28);
    // Custom limit is clamped to vramGB
    expect(effectiveVramGB('Apple M2 Ultra', 192, 256)).toBe(192);
  });

  it('unknown GPU name (e.g. Custom) returns vramGB unchanged', () => {
    expect(effectiveVramGB('Custom', 24)).toBe(24);
    expect(effectiveVramGB('MyGPU', 100)).toBe(100);
  });
});

describe('usableBytes with Apple wired limit', () => {
  it('applies Apple wired limit for Apple M2 Ultra (> 36 GB)', () => {
    const hw: HardwareSpec = {
      gpuName: 'Apple M2 Ultra',
      gpuCount: 1,
      vramGB: 192,
      bandwidthGBs: 800, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
    };
    // 192 * 0.75 * 0.95 * 1e9 = expected usable
    const expected = 192 * 0.75 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });

  it('applies Apple wired limit for Apple M4 Max (<= 36 GB)', () => {
    const hw: HardwareSpec = {
      gpuName: 'Apple M4 Max',
      gpuCount: 1,
      vramGB: 32,
      bandwidthGBs: 546, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
    };
    // 32 * 0.67 * 0.95 * 1e9 = expected usable
    const expected = 32 * 0.67 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });

  it('uses custom Apple wired limit when provided', () => {
    const hw: HardwareSpec = {
      gpuName: 'Apple M2 Ultra',
      gpuCount: 1,
      vramGB: 192,
      bandwidthGBs: 800, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
      appleWiredLimitGB: 120,
    };
    // 120 * 0.95 * 1e9 = expected usable
    const expected = 120 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });

  it('non-Apple GPUs ignore wired limit override', () => {
    const hw: HardwareSpec = {
      gpuName: 'H100 SXM',
      gpuCount: 1,
      vramGB: 80,
      bandwidthGBs: 3350, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
      appleWiredLimitGB: 50,
    };
    // Non-Apple GPU should use full 80 GB, ignoring the override
    const expected = 80 * 0.95 * 1e9;
    expect(usableBytes(hw)).toBeCloseTo(expected, 0);
  });
});

describe('maxUsers', () => {
  it('floors the free bytes over KV per request', () => {
    expect(maxUsers(100, 10, 30)).toBe(3);
    expect(maxUsers(100, 10, 90)).toBe(1);
  });
  it('is 0 when fixed memory meets or exceeds usable', () => {
    expect(maxUsers(100, 100, 1)).toBe(0);
    expect(maxUsers(100, 150, 1)).toBe(0);
  });
  it('is unbounded when KV per request is 0', () => {
    expect(maxUsers(100, 10, 0)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe('maxContext', () => {
  it('divides free bytes by users × bytes/token', () => {
    expect(maxContext(1000, 0, 2, 10, 1_000_000)).toBe(50);
    expect(maxContext(1000, 0, 0, 10, 1_000_000)).toBe(100); // users clamps to 1
  });
  it('is capped at maxPositionEmbeddings', () => {
    expect(maxContext(1e15, 0, 1, 1, 131072)).toBe(131072);
    expect(maxContext(1e9, 0, 1, 0, 4096)).toBe(4096);
  });
  it('is 0 when nothing is free', () => {
    expect(maxContext(10, 20, 1, 1, 4096)).toBe(0);
  });
});

describe('calculate', () => {
  it('fills every field for Llama 3 70B on 4× H100', () => {
    const r = calculate(state());
    expect(r.kvBytesPerToken).toBe(327_680);
    expect(r.kvBytesPerRequest).toBe(2_684_354_560);
    expect(r.kvBytesAllUsers).toBe(4 * 2_684_354_560);
    expect(r.weightBytes).toBe(141.2e9);
    expect(r.overheadBytes).toBe(4e9);
    expect(r.usableBytes).toBeCloseTo(304e9, 0);
    expect(r.totalBytes).toBeCloseTo(141.2e9 + 4e9 + 4 * 2_684_354_560, 0);
    expect(r.headroomBytes).toBeCloseTo(r.usableBytes - r.totalBytes, 0);
    expect(r.fits).toBe(true);
    expect(r.maxUsersAtContext).toBe(Math.floor((304e9 - 145.2e9) / 2_684_354_560));
    expect(r.maxContextForUsers).toBe(Math.min(131072, Math.floor((304e9 - 145.2e9) / (4 * 327_680))));
    // 4 GPUs: decode efficiency also carries the tensor-parallel communication penalty.
    expect(r.throughput.efficiency).toBeCloseTo(0.7 * tensorParallelEfficiency(4));
    expect(r.throughput.aggregateTokS).toBeCloseTo(r.throughput.perUserTokS * 4);
    // Prefill/TTFT: makeSpec() has no ffn, so the attention term falls back to hiddenSize.
    const expectedFlops = 2 * 70.6e9 * 8192 + 2 * 80 * 8192 * 8192 * 8192;
    expect(r.prefill.headsSource).toBe('hiddenSize-fallback');
    expect(r.prefill.mfu).toBe(0.4);
    expect(r.prefill.flops).toBe(expectedFlops);
    expect(r.prefill.ttftSeconds).toBeCloseTo(expectedFlops / (989.5e12 * 4 * 0.4), 6);
  });

  it('context table has 2K/8K/32K/128K, adding the chosen context sorted', () => {
    expect(calculate(state()).contextTable.map((r) => r.contextTokens)).toEqual([2048, 8192, 32768, 131072]);
    const t = calculate(state({ workload: { contextTokens: 16384, concurrentUsers: 1 } })).contextTable;
    expect(t.map((r) => r.contextTokens)).toEqual([2048, 8192, 16384, 32768, 131072]);
    expect(t[0].kvBytesPerRequest).toBe(671_088_640);
    expect(t[4].kvBytesPerRequest).toBe(42_949_672_960);
    expect(t[4].maxUsers).toBe(Math.floor((304e9 - 145.2e9) / 42_949_672_960));
  });

  it('does not fit when weights exceed VRAM: maxUsers 0, headroom negative', () => {
    const rtx4090: HardwareSpec = { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 165.0, reservePct: 5, overheadGB: 1 };
    const r = calculate(state({ hardware: rtx4090, workload: { contextTokens: 2048, concurrentUsers: 1 } }));
    expect(r.fits).toBe(false);
    expect(r.headroomBytes).toBeLessThan(0);
    expect(r.maxUsersAtContext).toBe(0);
    expect(r.maxContextForUsers).toBe(0);
    expect(r.contextTable.every((row) => row.maxUsers === 0)).toBe(true);
  });

  it('fits boundary: total exactly equal to usable fits', () => {
    // weights 20e9 − 4096 B, overhead 0, KV 4 B/token × 1024 tokens = 4096 B per user
    const model = makeSpec({ params: (20e9 - 4096) / 2, numLayers: 1, numKvHeads: 1, headDim: 1 });
    const hw: HardwareSpec = { gpuName: 'x', gpuCount: 1, vramGB: 20, bandwidthGBs: 100, tflopsBf16: 100, reservePct: 0, overheadGB: 0 };
    const one = calculate(state({ model, hardware: hw, workload: { contextTokens: 1024, concurrentUsers: 1 } }));
    expect(one.totalBytes).toBe(20e9);
    expect(one.usableBytes).toBe(20e9);
    expect(one.headroomBytes).toBe(0);
    expect(one.fits).toBe(true);
    expect(one.maxUsersAtContext).toBe(1);
    const two = calculate(state({ model, hardware: hw, workload: { contextTokens: 1024, concurrentUsers: 2 } }));
    expect(two.fits).toBe(false);
    expect(two.headroomBytes).toBe(-4096);
  });

  it('caps maxContext at maxPositionEmbeddings when memory is plentiful', () => {
    const b200x8: HardwareSpec = { gpuName: 'B200', gpuCount: 8, vramGB: 192, bandwidthGBs: 8000, tflopsBf16: 2250, reservePct: 5, overheadGB: 1 };
    const model = makeSpec({ params: 8e9, maxPositionEmbeddings: 8192 });
    expect(calculate(state({ model, hardware: b200x8 })).maxContextForUsers).toBe(8192);
  });

  it('throughput sanity: Llama 3.1 8B preset BF16 on a 4090 ≈ 40–45 tok/s', () => {
    const model = findModelPreset('Llama 3.1 8B');
    const gpu = findGpuPreset('RTX 4090');
    if (!model || !gpu) throw new Error('missing preset');
    const r = calculate({
      model,
      quant: { weight: 'bf16', kv: 'fp16' },
      hardware: { gpuName: gpu.name, gpuCount: 1, vramGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs, tflopsBf16: gpu.tflopsBf16, reservePct: 5, overheadGB: 1 },
      workload: { contextTokens: 2048, concurrentUsers: 1 },
      runtime: 'generic',
    });
    expect(r.throughput.perUserTokS).toBeGreaterThan(40);
    expect(r.throughput.perUserTokS).toBeLessThan(45);
  });

  it('tensor-parallel: single GPU is never flagged and KV is unscaled', () => {
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 65, tieEmbeddings: false } });
    const r = calculate(state({ model, hardware: { ...h100x4, gpuCount: 1 } }));
    expect(r.tensorParallel.headsDivisible).toBe(true);
    expect(r.tensorParallel.suggestedGpuCounts).toEqual([]);
    expect(r.tensorParallel.kvHeadsReplicated).toBe(false);
    expect(r.kvBytesPerToken).toBe(calculate(state({ model: makeSpec(), hardware: { ...h100x4, gpuCount: 1 } })).kvBytesPerToken);
  });

  it('tensor-parallel: flags a head count that does not split evenly and suggests valid counts', () => {
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 64, tieEmbeddings: false } });
    const r = calculate(state({ model, hardware: { ...h100x4, gpuCount: 3 } }));
    expect(r.tensorParallel.headsDivisible).toBe(false);
    expect(r.tensorParallel.suggestedGpuCounts).toEqual([1, 2, 4, 8]);
  });

  it('tensor-parallel: an evenly-splitting head count is not flagged', () => {
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 64, tieEmbeddings: false } });
    const r = calculate(state({ model, hardware: { ...h100x4, gpuCount: 4 } }));
    expect(r.tensorParallel.headsDivisible).toBe(true);
  });

  it('tensor-parallel: no ffn spec skips the head check without a false warning', () => {
    const model = makeSpec({ numKvHeads: 1 }); // no ffn; numKvHeads=1 splits/replicates evenly at any gpuCount
    const r = calculate(state({ model, hardware: { ...h100x4, gpuCount: 3 } }));
    expect(r.tensorParallel.checkable).toBe(false);
    expect(r.tensorParallel.headsDivisible).toBe(true);
    expect(r.tensorParallel.suggestedGpuCounts).toEqual([]);
  });

  it('tensor-parallel: fewer KV heads than GPUs replicates KV and scales the KV total', () => {
    const model = makeSpec({ numKvHeads: 2 });
    const r4 = calculate(state({ model, hardware: { ...h100x4, gpuCount: 4 } }));
    expect(r4.tensorParallel.kvHeadsReplicated).toBe(true);
    expect(r4.tensorParallel.effectiveKvHeads).toBe(4);
    expect(r4.tensorParallel.kvReplicationFactor).toBe(2);
    const r1 = calculate(state({ model, hardware: { ...h100x4, gpuCount: 1 } }));
    expect(r1.tensorParallel.kvHeadsReplicated).toBe(false);
    expect(r4.kvBytesPerToken).toBe(r1.kvBytesPerToken * 2);
    expect(r4.kvBytesPerRequest).toBe(r1.kvBytesPerRequest * 2);
  });

  it('tensor-parallel: MLA models are never KV-head-replicated, even with a small numKvHeads', () => {
    const model = makeSpec({ attention: 'mla', numKvHeads: 2, kvLoraRank: 512, qkRopeHeadDim: 64 });
    const r4 = calculate(state({ model, hardware: { ...h100x4, gpuCount: 4 } }));
    expect(r4.tensorParallel.kvHeadsReplicated).toBe(false);
    expect(r4.tensorParallel.kvHeadsSplitValid).toBe(true);
    expect(r4.tensorParallel.kvReplicationFactor).toBe(1);
    const r1 = calculate(state({ model, hardware: { ...h100x4, gpuCount: 1 } }));
    expect(r4.kvBytesPerToken).toBe(r1.kvBytesPerToken);
  });

  it('MoE throughput reads only active weights', () => {
    const dense = makeSpec({ params: 30e9 });
    const moe = makeSpec({ params: 30e9, moe: { numExperts: 128, expertsPerToken: 8, sharedExperts: 0 } });
    const a = calculate(state({ model: dense })).throughput.perUserTokS;
    const b = calculate(state({ model: moe })).throughput.perUserTokS;
    expect(b).toBeGreaterThan(a);
    expect(calculate(state({ model: moe })).weightBytes).toBe(60e9); // all experts resident
  });
});

describe('runtime profiles: generic is bit-identical to today', () => {
  it('usableBytesForRuntime/overheadBytesForRuntime match the plain (pre-runtime) formulas for generic', () => {
    expect(usableBytesForRuntime(h100x4, 'generic')).toBe(usableBytes(h100x4));
    expect(overheadBytesForRuntime(h100x4, 'generic')).toBe(overheadBytes(h100x4));
    // llama.cpp and MLX reuse the same generic accounting (no reserve-% or overhead override).
    expect(usableBytesForRuntime(h100x4, 'llamacpp')).toBe(usableBytes(h100x4));
    expect(usableBytesForRuntime(h100x4, 'mlx')).toBe(usableBytes(h100x4));
  });

  it("calculate() with runtime: 'generic' produces every number the pre-runtime calculator did", () => {
    const s = state();
    const r = calculate(s);
    expect(r.usableBytes).toBe(usableBytes(s.hardware));
    expect(r.overheadBytes).toBe(overheadBytes(s.hardware));
    expect(r.kvBytesPerRequest).toBe(r.kvBytesPerToken * s.workload.contextTokens);
  });
});

describe('runtime profiles: vLLM usable-memory maths', () => {
  it('usable = gpuCount × effectiveVramGB × 1e9 × gpu_memory_utilization, ignoring reservePct', () => {
    const hw: HardwareSpec = { ...h100x4, reservePct: 40 }; // a high reserve % must have no effect under vLLM
    expect(usableBytesForRuntime(hw, 'vllm')).toBeCloseTo(hw.gpuCount * hw.vramGB * 1e9 * VLLM_GPU_MEMORY_UTILIZATION, 0);
  });

  it('adds a flat CUDA-graph/activation overhead allowance on top of the generic overhead', () => {
    expect(overheadBytesForRuntime(h100x4, 'vllm')).toBe(overheadBytes(h100x4) + VLLM_OVERHEAD_ALLOWANCE_GB * 1e9 * h100x4.gpuCount);
  });

  it('rounds the KV per request up to a 16-token block multiple', () => {
    const withOddContext = calculate(state({ runtime: 'vllm', workload: { contextTokens: 8193, concurrentUsers: 1 } }));
    const withRoundedContext = calculate(state({ runtime: 'generic', workload: { contextTokens: 8208, concurrentUsers: 1 } }));
    expect(withOddContext.kvBytesPerRequest).toBe(withRoundedContext.kvBytesPerRequest);
  });

  it('does not round a context that is already a block multiple', () => {
    const exact = calculate(state({ runtime: 'vllm', workload: { contextTokens: 8192, concurrentUsers: 1 } }));
    const generic = calculate(state({ runtime: 'generic', workload: { contextTokens: 8192, concurrentUsers: 1 } }));
    expect(exact.kvBytesPerRequest).toBe(generic.kvBytesPerRequest);
  });
});

describe('runtime profiles: vLLM max-context block rounding', () => {
  it('reports a context whose actual (block-rounded) KV reservation still fits under usable (repro: Llama 3.1 8B, H100 80GB, reserve 5%, overhead 1GB, N=4)', () => {
    const model = findModelPreset('Llama 3.1 8B');
    if (!model) throw new Error('missing preset');
    const hardware: HardwareSpec = { gpuName: 'H100 SXM', gpuCount: 1, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 100, reservePct: 5, overheadGB: 1 };
    // workload.contextTokens does not affect maxContextForUsers; any value probes it.
    const probe = calculate(state({ model, hardware, runtime: 'vllm', workload: { contextTokens: 8192, concurrentUsers: 4 } }));
    const C = probe.maxContextForUsers;
    expect(C).toBeLessThan(model.maxPositionEmbeddings); // memory-bound in this scenario, not the model cap
    expect(C % VLLM_KV_BLOCK_TOKENS).toBe(0);
    const atC = calculate(state({ model, hardware, runtime: 'vllm', workload: { contextTokens: C, concurrentUsers: 4 } }));
    expect(atC.totalBytes).toBeLessThanOrEqual(atC.usableBytes);
  });

  it('never overshoots usable across a range of models, hardware and user counts (that fit at all)', () => {
    const llama8b = findModelPreset('Llama 3.1 8B');
    if (!llama8b) throw new Error('missing preset');
    const configs: Array<{ model: ModelSpec; hardware: HardwareSpec; users: number }> = [
      { model: makeSpec(), hardware: h100x4, users: 4 }, // 70B dense, comfortably fits 4×H100
      { model: llama8b, hardware: { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 100, reservePct: 5, overheadGB: 1 }, users: 8 },
      {
        model: makeSpec({ numKvHeads: 32, headDim: 128, numLayers: 40 }),
        hardware: { gpuName: 'H200', gpuCount: 2, vramGB: 141, bandwidthGBs: 4800, tflopsBf16: 100, reservePct: 10, overheadGB: 2 },
        users: 16,
      },
    ];
    for (const { model, hardware, users } of configs) {
      const probe = calculate(state({ model, hardware, runtime: 'vllm', workload: { contextTokens: 4096, concurrentUsers: users } }));
      const C = probe.maxContextForUsers;
      // Sanity check the fixture: the model must actually fit before any context is asked for.
      expect(probe.weightBytes + probe.overheadBytes).toBeLessThan(probe.usableBytes);
      const atC = calculate(state({ model, hardware, runtime: 'vllm', workload: { contextTokens: C, concurrentUsers: users } }));
      expect(atC.totalBytes).toBeLessThanOrEqual(atC.usableBytes);
    }
  });

  it('non-vLLM runtimes keep the unrounded maxContext formula', () => {
    for (const runtime of ['generic', 'llamacpp', 'sglang', 'mlx'] as const) {
      expect(maxContext(1000, 0, 2, 10, 1_000_000, runtime)).toBe(50);
    }
    expect(maxContext(1000, 0, 2, 10, 1_000_000, 'vllm')).toBe(48); // 50 rounded down to a 16-token block
  });
});

describe('runtime profiles: SGLang usable-memory maths', () => {
  it('usable = gpuCount × effectiveVramGB × 1e9 × mem_fraction_static, ignoring reservePct', () => {
    const hw: HardwareSpec = { ...h100x4, reservePct: 40 };
    expect(usableBytesForRuntime(hw, 'sglang')).toBeCloseTo(hw.gpuCount * hw.vramGB * 1e9 * SGLANG_MEM_FRACTION_STATIC, 0);
    expect(overheadBytesForRuntime(hw, 'sglang')).toBe(overheadBytes(hw)); // no extra allowance
  });
});

describe('runtime profiles: MLX builds on the Apple wired-memory limit, not a duplicate', () => {
  it('an Apple GPU gets the same effective VRAM under MLX as under generic', () => {
    const apple: HardwareSpec = { gpuName: 'Apple M2 Ultra', gpuCount: 1, vramGB: 192, bandwidthGBs: 800, tflopsBf16: 100, reservePct: 5, overheadGB: 1 };
    expect(usableBytesForRuntime(apple, 'mlx')).toBe(usableBytesForRuntime(apple, 'generic'));
    expect(effectiveVramGB(apple.gpuName, apple.vramGB)).toBeLessThan(apple.vramGB); // wired limit applied
  });
});

describe('CPU/RAM offload (#7)', () => {
  it('offload off (no offload field): every number matches a pre-offload calculation bit-for-bit', () => {
    const model = makeSpec({ params: 200e9 }); // too big for a single 24 GB GPU, on purpose
    const rtx4090: HardwareSpec = { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 100, reservePct: 5, overheadGB: 1 };
    const withoutField = calculate(state({ model, hardware: rtx4090 }));
    const withDisabled = calculate(
      state({ model, hardware: { ...rtx4090, offload: { enabled: false, systemRamGB: 64, ramBandwidthGBs: 50 } } }),
    );
    expect(withDisabled.fits).toBe(withoutField.fits);
    expect(withDisabled.headroomBytes).toBe(withoutField.headroomBytes);
    expect(withDisabled.totalBytes).toBe(withoutField.totalBytes);
    expect(withDisabled.throughput).toEqual(withoutField.throughput);
    expect(withDisabled.throughput.perUserTokS).toBe(withoutField.throughput.perUserTokS);
    expect(withDisabled.maxUsersAtContext).toBe(withoutField.maxUsersAtContext);
    expect(withDisabled.maxContextForUsers).toBe(withoutField.maxContextForUsers);
    expect(withDisabled.contextTable).toEqual(withoutField.contextTable);
    // The offload plan is a no-op split (everything on the GPU) when disabled.
    expect(withoutField.offload.gpuLayers).toBe(model.numLayers);
    expect(withoutField.offload.cpuLayers).toBe(0);
    // fixedBytes/bytesPerUser (shared with the chart and other panels) must equal the un-split
    // weights+overhead and KV-per-request exactly when offload is off — bit-identical, not just close.
    expect(withoutField.fixedBytes).toBe(withoutField.weightBytes + withoutField.overheadBytes);
    expect(withoutField.bytesPerUser).toBe(withoutField.kvBytesPerRequest);
    expect(withDisabled.fixedBytes).toBe(withoutField.fixedBytes);
    expect(withDisabled.bytesPerUser).toBe(withoutField.bytesPerUser);
  });

  it('offload on: a 70B Q4 model that does not fit on one 24 GB GPU offloads layers to RAM and still reports a throughput', () => {
    const model = makeSpec(); // Llama 3 70B defaults
    const rtx4090: HardwareSpec = {
      gpuName: 'RTX 4090',
      gpuCount: 1,
      vramGB: 24,
      bandwidthGBs: 1008, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
      offload: { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 },
    };
    const r = calculate(
      state({ model, quant: { weight: 'q4_k_m', kv: 'fp16' }, hardware: rtx4090, workload: { contextTokens: 2048, concurrentUsers: 1 } }),
    );
    expect(r.offload.cpuLayers).toBeGreaterThan(0);
    expect(r.offload.gpuLayers).toBeLessThan(model.numLayers);
    expect(r.offload.fitsInRam).toBe(true);
    expect(r.throughput.perUserTokS).toBeGreaterThan(0);
    // Weights alone (~42.8 GB at Q4_K_M) still don't fit the GPU's ~21.8 GB usable on their own, offload or not.
    expect(r.fits).toBe(false);
    // The issue's own worked example: -ngl 39/80 at ~1.52 tok/s.
    expect(r.offload.gpuLayers).toBe(39);
    expect(r.offload.cpuLayers).toBe(41);
    expect(r.throughput.perUserTokS).toBeCloseTo(1.52, 2);
    // Capacity math must agree with the "Offloaded" badge, not the un-split "does not fit":
    // the GPU-resident weights (not the full 42.8 GB) are what's fixed once offload is on.
    expect(r.maxUsersAtContext).toBeGreaterThanOrEqual(1);
    expect(r.fixedBytes).toBe(r.offload.gpuWeightBytes + r.overheadBytes);
    expect(r.bytesPerUser).toBe(r.kvBytesPerRequest);
    // What Chart.tsx plots for "current users" must actually clear usable VRAM (a green marker),
    // matching the Offloaded (not does-not-fit) badge.
    expect(r.fixedBytes + r.bytesPerUser * 1).toBeLessThanOrEqual(r.usableBytes);
  });

  it('offload on: does not fit even with RAM when system RAM is too small', () => {
    const model = makeSpec();
    const rtx4090: HardwareSpec = {
      gpuName: 'RTX 4090',
      gpuCount: 1,
      vramGB: 24,
      bandwidthGBs: 1008, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
      offload: { enabled: true, systemRamGB: 1, ramBandwidthGBs: 50 },
    };
    const r = calculate(state({ model, hardware: rtx4090, workload: { contextTokens: 2048, concurrentUsers: 1 } }));
    expect(r.offload.cpuLayers).toBeGreaterThan(0);
    expect(r.offload.fitsInRam).toBe(false);
  });

  it('offload on: KV + overhead alone exceed usable VRAM -> does not fit, even with unlimited system RAM', () => {
    // 1x RTX 4090, 8 users at 32K: KV for all users alone is already far more than usable VRAM.
    const model = makeSpec();
    const rtx4090: HardwareSpec = {
      gpuName: 'RTX 4090',
      gpuCount: 1,
      vramGB: 24,
      bandwidthGBs: 1008, tflopsBf16: 100,
      reservePct: 5,
      overheadGB: 1,
      offload: { enabled: true, systemRamGB: 1_000_000, ramBandwidthGBs: 50 },
    };
    const r = calculate(state({ model, hardware: rtx4090, workload: { contextTokens: 32768, concurrentUsers: 8 } }));
    expect(r.offload.cpuLayers).toBeGreaterThan(0);
    // Even with unlimited RAM, the GPU alone can't hold its own KV cache and overhead.
    expect(r.offload.fitsInRam).toBe(false);
    // A configuration that doesn't actually run must not present a throughput as achievable.
    expect(r.throughput.perUserTokS).toBe(0);
    expect(r.throughput.aggregateTokS).toBe(0);
  });
});
