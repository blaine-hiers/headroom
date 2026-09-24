import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { buildMarkdownSummary } from './exportMarkdown';
import { calculate } from './fit';
import { DISABLED_SPECULATIVE } from './speculative';
import type { CalcState, HardwareSpec } from './types';

const h100x4: HardwareSpec = { gpuName: 'H100 SXM', gpuCount: 4, vramGB: 80, bandwidthGBs: 3350, tflopsBf16: 989.5, reservePct: 5, overheadGB: 1 };

function state(overrides: Partial<CalcState> = {}): CalcState {
  return {
    model: makeSpec(),
    quant: { weight: 'bf16', kv: 'fp16' },
    hardware: h100x4,
    workload: { contextTokens: 8192, concurrentUsers: 4 },
    runtime: 'generic',
    speculative: DISABLED_SPECULATIVE,
    ...overrides,
  };
}

const LINK = 'https://blaine-hiers.github.io/headroom/?m=test';

describe('buildMarkdownSummary', () => {
  it('includes the model, quant, hardware, workload and the share link', () => {
    const s = state();
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toContain('## Headroom: Llama 3 70B');
    expect(md).toContain('- **Model:** test/llama-3-70b');
    expect(md).toContain('weight BF16, KV FP16');
    expect(md).toContain('4 × H100 SXM');
    expect(md).toContain('4 users @ 8K context');
    expect(md).toContain(`[Open this configuration in Headroom](${LINK})`);
  });

  it('reports Fits with the headroom amount when the model fits', () => {
    const s = state();
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toMatch(/\*\*Fits\*\* — .+ headroom/);
  });

  it('reports Does not fit with the shortfall when it does not', () => {
    const s = state({ hardware: { ...h100x4, gpuCount: 1, vramGB: 8 } });
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toMatch(/\*\*Does not fit\*\* — .+ short/);
  });

  it('renders the context table with a header and one row per context', () => {
    const s = state();
    const result = calculate(s);
    const md = buildMarkdownSummary(s, result, LINK);
    expect(md).toContain('### Context table');
    expect(md).toContain('| Context | KV per request | Max users |');
    const tableRows = md.split('### Context table')[1].split('\n').filter((l) => l.startsWith('| ') && !l.includes('---') && !l.includes('Context |'));
    expect(tableRows).toHaveLength(result.contextTable.length);
  });

  it('omits the cloud cost row when no $/GPU-hour is set', () => {
    const s = state();
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).not.toContain('Cloud cost');
  });

  it('includes the cloud cost row when a $/GPU-hour price is set', () => {
    const s = state({ hardware: { ...h100x4, usdPerHour: 3.25 } });
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toContain('| Cloud cost |');
    expect(md).toContain('/hr');
  });

  it('omits the CPU offload row when offload is off or a no-op', () => {
    const s = state();
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).not.toContain('CPU offload');
  });

  it('includes the CPU offload row once layers actually spill to system RAM', () => {
    const s = state({
      hardware: { ...h100x4, gpuCount: 1, vramGB: 40, offload: { enabled: true, systemRamGB: 512, ramBandwidthGBs: 90 } },
    });
    const result = calculate(s);
    expect(result.offload.cpuLayers).toBeGreaterThan(0);
    const md = buildMarkdownSummary(s, result, LINK);
    expect(md).toContain('| CPU offload | -ngl');
  });

  it('omits the speculative-decoding row when disabled', () => {
    const s = state();
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).not.toContain('Speculative decoding');
  });

  it('includes the speculative-decoding multiplier when enabled', () => {
    const s = state({ speculative: { ...DISABLED_SPECULATIVE, enabled: true, draftMode: 'none' } });
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toMatch(/\| Speculative decoding \| ×[\d.]+ vs\. no speculation \|/);
  });

  it('labels file-sourced weights with their source instead of "estimated"', () => {
    const s = state({ model: makeSpec({ fileWeights: { bytes: 4e10, label: 'Q4_K_M GGUF', quant: 'q4_k_m' } }), quant: { weight: 'q4_k_m', kv: 'fp16' } });
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toContain('Q4_K_M GGUF, from repo files');
  });

  it('escapes a "|" in a file-weights label so it cannot split the table into extra columns', () => {
    const s = state({
      model: makeSpec({ fileWeights: { bytes: 4e10, label: 'weird|file|name.gguf', quant: 'q4_k_m' } }),
      quant: { weight: 'q4_k_m', kv: 'fp16' },
    });
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    const weightsRow = md.split('\n').find((l) => l.startsWith('| Weights |'));
    expect(weightsRow).toBeDefined();
    // Exactly the table's own two separators ("| Weights |" and the trailing "|") should remain
    // unescaped; every "|" that came from the label must be escaped.
    expect(weightsRow).toContain('weird\\|file\\|name.gguf');
    expect(weightsRow!.match(/(?<!\\)\|/g)).toHaveLength(3);
  });

  it('strips a newline out of a file-weights label so it cannot break the table', () => {
    const s = state({
      model: makeSpec({ fileWeights: { bytes: 4e10, label: 'multi\nline label', quant: 'q4_k_m' } }),
      quant: { weight: 'q4_k_m', kv: 'fp16' },
    });
    const md = buildMarkdownSummary(s, calculate(s), LINK);
    expect(md).toContain('multi line label');
    expect(md).not.toContain('multi\nline label');
  });
});
