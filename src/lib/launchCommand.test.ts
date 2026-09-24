import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { calculate } from './fit';
import { buildLaunchCommand, shellQuote, vllmCpuOffloadGiB } from './launchCommand';
import { DISABLED_SPECULATIVE } from './speculative';
import type { CalcState } from './types';

const hardware: CalcState['hardware'] = { gpuName: 'H100 SXM', gpuCount: 1, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 100, reservePct: 5, overheadGB: 1 };
const appleHardware: CalcState['hardware'] = { gpuName: 'Apple M2 Ultra', gpuCount: 1, vramGB: 192, bandwidthGBs: 800, tflopsBf16: 100, reservePct: 5, overheadGB: 1 };

function state(overrides: Partial<CalcState> = {}): CalcState {
  return {
    model: makeSpec({ id: 'meta-llama/Llama-3.3-70B-Instruct' }),
    quant: { weight: 'bf16', kv: 'fp16' },
    hardware,
    workload: { contextTokens: 8192, concurrentUsers: 4 },
    runtime: 'generic',
    ...overrides,
  };
}

describe('shellQuote', () => {
  it('leaves a safe argument (repo id, path, number) unquoted', () => {
    expect(shellQuote('meta-llama/Llama-3.3-70B-Instruct')).toBe('meta-llama/Llama-3.3-70B-Instruct');
    expect(shellQuote('/path/to/model.gguf')).toBe('/path/to/model.gguf');
  });

  it('single-quotes an argument with a space or shell metacharacter', () => {
    expect(shellQuote('org/model name')).toBe("'org/model name'");
    expect(shellQuote('org/model; rm -rf /')).toBe("'org/model; rm -rf /'");
  });

  it('escapes an embedded single quote with the POSIX close-escape-reopen idiom', () => {
    expect(shellQuote("org/it's a model")).toBe("'org/it'\\''s a model'");
  });
});

describe('buildLaunchCommand', () => {
  it('generic has no launch command', () => {
    expect(buildLaunchCommand(state({ runtime: 'generic' }))).toEqual({ command: null, notes: [] });
  });

  it('vLLM: single GPU, default KV dtype — omits tensor-parallel-size and kv-cache-dtype', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'vllm' }));
    expect(command).toBe('vllm serve meta-llama/Llama-3.3-70B-Instruct --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.9');
  });

  it('vLLM: multi-GPU and FP8 KV cache adds tensor-parallel-size and kv-cache-dtype', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'vllm', hardware: { ...hardware, gpuCount: 4 }, quant: { weight: 'bf16', kv: 'fp8' } }));
    expect(command).toBe(
      'vllm serve meta-llama/Llama-3.3-70B-Instruct --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --tensor-parallel-size 4',
    );
  });

  it('vLLM: int8 KV cache approximates to fp8 and adds a caveat note', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'vllm', quant: { weight: 'bf16', kv: 'int8' } }));
    expect(command).toContain('--kv-cache-dtype fp8');
    expect(notes.some((n) => n.includes('INT8'))).toBe(true);
  });

  it('vLLM: shell-quotes a model id with a space and a semicolon', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'vllm', model: makeSpec({ id: 'org/my model; rm -rf /' }) }));
    expect(command).toContain("'org/my model; rm -rf /'");
  });

  it('vLLM on an Apple GPU adds a hardware-mismatch note', () => {
    const { notes } = buildLaunchCommand(state({ runtime: 'vllm', hardware: appleHardware }));
    expect(notes.some((n) => n.includes('NVIDIA/AMD'))).toBe(true);
  });

  it('llama.cpp: a GGUF-looking repo id keeps -hf', () => {
    const ggufId = 'bartowski/Llama-3.3-70B-Instruct-GGUF';
    const { command, notes } = buildLaunchCommand(state({ runtime: 'llamacpp', model: makeSpec({ id: ggufId }), workload: { contextTokens: 8192, concurrentUsers: 1 } }));
    expect(command).toBe(`llama-server -hf ${ggufId} -c 8192 -ngl 999`);
    expect(notes.some((n) => n.includes('needs a GGUF file'))).toBe(false);
  });

  it('llama.cpp: a non-GGUF (safetensors) repo id falls back to -m with a GGUF caveat note', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'llamacpp', workload: { contextTokens: 8192, concurrentUsers: 1 } }));
    expect(command).toBe('llama-server -m /path/to/model.gguf -c 8192 -ngl 999');
    expect(notes.some((n) => n.includes('needs a GGUF file'))).toBe(true);
  });

  it('llama.cpp: single user omits -np, and default KV cache type omits cache-type flags', () => {
    const { notes } = buildLaunchCommand(state({ runtime: 'llamacpp', workload: { contextTokens: 8192, concurrentUsers: 1 } }));
    expect(notes.some((n) => n.includes('below the chosen context'))).toBe(false);
  });

  it('llama.cpp: multiple users multiplies -c by -np and stays evenly split (no warning)', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'llamacpp', workload: { contextTokens: 4096, concurrentUsers: 4 } }));
    expect(command).toContain('-c 16384');
    expect(command).toContain('-np 4');
    expect(notes.some((n) => n.includes('below the chosen context'))).toBe(false);
  });

  it('llama.cpp: non-default KV quant sets --cache-type-k/-v', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'llamacpp', model: makeSpec({ id: 'bartowski/Llama-3.3-70B-Instruct-GGUF' }), quant: { weight: 'bf16', kv: 'fp8' } }));
    expect(command).toContain('--cache-type-k q8_0');
    expect(command).toContain('--cache-type-v q8_0');
  });

  it('llama.cpp: shell-quotes a GGUF repo id with a space and a semicolon', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'llamacpp', model: makeSpec({ id: 'org/my gguf; rm -rf /' }) }));
    expect(command).toContain("-hf 'org/my gguf; rm -rf /'");
  });

  it('SGLang: default fraction and single GPU', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'sglang' }));
    expect(command).toBe(
      'python -m sglang.launch_server --model-path meta-llama/Llama-3.3-70B-Instruct --context-length 8192 --max-running-requests 4 --mem-fraction-static 0.88',
    );
  });

  it('SGLang: multi-GPU adds --tp', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'sglang', hardware: { ...hardware, gpuCount: 2 } }));
    expect(command).toContain('--tp 2');
  });

  it('SGLang: shell-quotes a model id with a space and a semicolon', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'sglang', model: makeSpec({ id: 'org/my model; rm -rf /' }) }));
    expect(command).toContain("--model-path 'org/my model; rm -rf /'");
  });

  it('SGLang on an Apple GPU adds a hardware-mismatch note', () => {
    const { notes } = buildLaunchCommand(state({ runtime: 'sglang', hardware: appleHardware }));
    expect(notes.some((n) => n.includes('NVIDIA/AMD'))).toBe(true);
  });

  it('MLX: builds a command from the model id and context, with unified-memory notes', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'mlx' }));
    expect(command).toBe('mlx_lm.server --model meta-llama/Llama-3.3-70B-Instruct --max-kv-size 8192');
    expect(notes.length).toBeGreaterThan(0);
  });

  it('MLX: shell-quotes a model id with a space and a semicolon', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'mlx', model: makeSpec({ id: 'org/my model; rm -rf /' }) }));
    expect(command).toContain("--model 'org/my model; rm -rf /'");
  });

  it('MLX on a non-Apple GPU adds a hardware-mismatch note; on Apple hardware it does not', () => {
    const onH100 = buildLaunchCommand(state({ runtime: 'mlx', hardware }));
    expect(onH100.notes.some((n) => n.includes('Apple silicon'))).toBe(true);
    const onApple = buildLaunchCommand(state({ runtime: 'mlx', hardware: appleHardware }));
    expect(onApple.notes.some((n) => n.includes('Apple silicon'))).toBe(false);
  });

  it('an unknown/custom GPU name is not flagged as a mismatch for any runtime', () => {
    const custom: CalcState['hardware'] = { ...hardware, gpuName: 'Custom' };
    for (const runtime of ['vllm', 'llamacpp', 'sglang', 'mlx'] as const) {
      const { notes } = buildLaunchCommand(state({ runtime, hardware: custom }));
      expect(notes.some((n) => n.includes('NVIDIA/AMD') || n.includes('Apple silicon'))).toBe(false);
    }
  });

  it('every non-generic runtime flags the command as a starting point', () => {
    for (const runtime of ['vllm', 'llamacpp', 'sglang', 'mlx'] as const) {
      const { notes } = buildLaunchCommand(state({ runtime }));
      expect(notes.some((n) => n.toLowerCase().includes('starting point'))).toBe(true);
    }
  });

  it('matches a snapshot per runtime', () => {
    const commands = (['generic', 'vllm', 'llamacpp', 'sglang', 'mlx'] as const).map((runtime) => buildLaunchCommand(state({ runtime })));
    expect(commands).toMatchSnapshot();
  });
});

describe('buildLaunchCommand with CPU/RAM offload and speculative decoding (#20)', () => {
  const rtx4090: CalcState['hardware'] = {
    gpuName: 'RTX 4090',
    gpuCount: 1,
    vramGB: 24,
    bandwidthGBs: 1008,
    tflopsBf16: 165,
    reservePct: 5,
    overheadGB: 1,
    offload: { enabled: true, systemRamGB: 64, ramBandwidthGBs: 50 },
  };
  const offloaded = (overrides: Partial<CalcState> = {}) =>
    state({
      model: makeSpec({ id: 'bartowski/Llama-3.3-70B-Instruct-GGUF' }),
      quant: { weight: 'q4_k_m', kv: 'fp16' },
      hardware: rtx4090,
      workload: { contextTokens: 2048, concurrentUsers: 1 },
      ...overrides,
    });

  it('llama.cpp: emits the computed -ngl (matching the verdict), not 999, and no "whole model" note', () => {
    const s = offloaded({ runtime: 'llamacpp' });
    const plan = calculate(s).offload;
    expect(plan.gpuLayers).toBe(39);
    const { command, notes } = buildLaunchCommand(s);
    expect(command).toBe('llama-server -hf bartowski/Llama-3.3-70B-Instruct-GGUF -c 2048 -ngl 39');
    expect(notes.join(' ')).not.toMatch(/-ngl 999|whole model/);
    expect(notes[0]).toMatch(/-ngl 39 keeps 39 of 80 layers on the GPU/);
  });

  it('llama.cpp: offload on but nothing spills keeps -ngl 999 so the output layer stays on the GPU', () => {
    const s = offloaded({ runtime: 'llamacpp', model: makeSpec({ id: 'bartowski/Llama-3.1-8B-Instruct-GGUF', params: 8e9, numLayers: 32 }) });
    expect(calculate(s).offload.cpuLayers).toBe(0);
    const { command, notes } = buildLaunchCommand(s);
    expect(command).toMatch(/-ngl 999$/);
    expect(notes[0]).toMatch(/-ngl 999/);
  });

  it('llama.cpp: offload OFF still emits -ngl 999 with its note', () => {
    const s = offloaded({ runtime: 'llamacpp', hardware: { ...rtx4090, offload: { enabled: false, systemRamGB: 64, ramBandwidthGBs: 50 } } });
    const { command, notes } = buildLaunchCommand(s);
    expect(command).toBe('llama-server -hf bartowski/Llama-3.3-70B-Instruct-GGUF -c 2048 -ngl 999');
    expect(notes[0]).toMatch(/-ngl 999/);
  });

  it('llama.cpp: a split that does not run says so', () => {
    const s = offloaded({ runtime: 'llamacpp', hardware: { ...rtx4090, offload: { enabled: true, systemRamGB: 1, ramBandwidthGBs: 50 } } });
    expect(buildLaunchCommand(s).notes.join(' ')).toMatch(/does not run even with CPU\/RAM offload/);
  });

  it('vLLM: adds --cpu-offload-gb (GiB per GPU of RAM-resident weights, rounded up)', () => {
    for (const gpuCount of [1, 2]) {
      const s = offloaded({ runtime: 'vllm', hardware: { ...rtx4090, gpuCount } });
      const plan = calculate(s).offload;
      expect(plan.cpuWeightBytes).toBeGreaterThan(0);
      const gib = vllmCpuOffloadGiB(plan.cpuWeightBytes, gpuCount);
      expect(gib).toBe(Math.ceil(plan.cpuWeightBytes / gpuCount / 2 ** 30));
      const { command, notes } = buildLaunchCommand(s);
      expect(command).toMatch(new RegExp(`--cpu-offload-gb ${gib}( |$)`));
      expect(notes.join(' ')).toMatch(/--cpu-offload-gb is GiB per GPU/);
    }
  });

  it('vLLM: offload off (or nothing spilled) adds no --cpu-offload-gb', () => {
    const off = offloaded({ runtime: 'vllm', hardware: { ...rtx4090, offload: { enabled: false, systemRamGB: 64, ramBandwidthGBs: 50 } } });
    expect(buildLaunchCommand(off).command).not.toMatch(/cpu-offload-gb/);
    const fitsAnyway = offloaded({ runtime: 'vllm', model: makeSpec({ id: 'org/small', params: 8e9 }) });
    expect(calculate(fitsAnyway).offload.cpuLayers).toBe(0);
    expect(buildLaunchCommand(fitsAnyway).command).not.toMatch(/cpu-offload-gb/);
  });

  it('vLLM: speculative decoding on adds a note (no invented flags); off adds nothing', () => {
    const on = offloaded({ runtime: 'vllm', speculative: { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'none' } });
    const off = offloaded({ runtime: 'vllm', speculative: DISABLED_SPECULATIVE });
    expect(buildLaunchCommand(on).notes.join(' ')).toMatch(/configure the draft model separately/);
    expect(buildLaunchCommand(on).command).toBe(buildLaunchCommand(off).command);
    expect(buildLaunchCommand(off).notes.join(' ')).not.toMatch(/draft model/);
  });

  it('SGLang: offload on adds a "not modelled" note and no flag', () => {
    const on = offloaded({ runtime: 'sglang' });
    const off = offloaded({ runtime: 'sglang', hardware: { ...rtx4090, offload: { enabled: false, systemRamGB: 64, ramBandwidthGBs: 50 } } });
    expect(buildLaunchCommand(on).command).toBe(buildLaunchCommand(off).command);
    expect(buildLaunchCommand(on).notes.join(' ')).toMatch(/CPU\/RAM offload is not modelled for SGLang/);
    expect(buildLaunchCommand(off).notes.join(' ')).not.toMatch(/offload/);
  });
});
