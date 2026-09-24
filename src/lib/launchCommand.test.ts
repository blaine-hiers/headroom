import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { buildLaunchCommand } from './launchCommand';
import type { CalcState } from './types';

const hardware: CalcState['hardware'] = { gpuName: 'H100 SXM', gpuCount: 1, vramGB: 80, bandwidthGBs: 3350, reservePct: 5, overheadGB: 1 };

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

  it('llama.cpp: single user omits -np, and default KV cache type omits cache-type flags', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'llamacpp', workload: { contextTokens: 8192, concurrentUsers: 1 } }));
    expect(command).toBe('llama-server -hf meta-llama/Llama-3.3-70B-Instruct -c 8192 -ngl 999');
    expect(notes.some((n) => n.includes('below the chosen context'))).toBe(false);
  });

  it('llama.cpp: multiple users multiplies -c by -np and stays evenly split (no warning)', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'llamacpp', workload: { contextTokens: 4096, concurrentUsers: 4 } }));
    expect(command).toContain('-c 16384');
    expect(command).toContain('-np 4');
    expect(notes.some((n) => n.includes('below the chosen context'))).toBe(false);
  });

  it('llama.cpp: non-default KV quant sets --cache-type-k/-v', () => {
    const { command } = buildLaunchCommand(state({ runtime: 'llamacpp', quant: { weight: 'bf16', kv: 'fp8' } }));
    expect(command).toContain('--cache-type-k q8_0');
    expect(command).toContain('--cache-type-v q8_0');
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

  it('MLX: builds a command from the model id and context, with unified-memory notes', () => {
    const { command, notes } = buildLaunchCommand(state({ runtime: 'mlx' }));
    expect(command).toBe('mlx_lm.server --model meta-llama/Llama-3.3-70B-Instruct --max-kv-size 8192');
    expect(notes.length).toBeGreaterThan(0);
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
