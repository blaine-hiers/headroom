import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { buildLaunchCommand, shellQuote } from './launchCommand';
import type { CalcState } from './types';

const hardware: CalcState['hardware'] = { gpuName: 'H100 SXM', gpuCount: 1, vramGB: 80, bandwidthGBs: 3350, reservePct: 5, overheadGB: 1 };
const appleHardware: CalcState['hardware'] = { gpuName: 'Apple M2 Ultra', gpuCount: 1, vramGB: 192, bandwidthGBs: 800, reservePct: 5, overheadGB: 1 };

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
