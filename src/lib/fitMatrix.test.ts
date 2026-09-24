import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { calculate } from './fit';
import { FIT_MATRIX_QUANTS, fitMatrix, fitMatrixContexts } from './fitMatrix';
import { WEIGHT_QUANTS } from './quant';
import type { CalcState, HardwareSpec } from './types';

const hw: HardwareSpec = { gpuName: 'RTX 4090', gpuCount: 1, vramGB: 24, bandwidthGBs: 1008, tflopsBf16: 82.6, reservePct: 5, overheadGB: 1 };

function state(overrides: Partial<CalcState> = {}): CalcState {
  return {
    model: makeSpec(),
    quant: { weight: 'bf16', kv: 'fp16' },
    hardware: hw,
    workload: { contextTokens: 8192, concurrentUsers: 4 },
    runtime: 'generic',
    ...overrides,
  };
}

describe('fitMatrixContexts', () => {
  it('includes the standard table contexts, maxPositionEmbeddings, and the chosen context', () => {
    const model = makeSpec({ maxPositionEmbeddings: 131072 });
    expect(fitMatrixContexts(model, 8192)).toEqual([2048, 8192, 32768, 131072]);
  });

  it('adds the chosen context when it differs from every other column', () => {
    const model = makeSpec({ maxPositionEmbeddings: 131072 });
    expect(fitMatrixContexts(model, 16384)).toEqual([2048, 8192, 16384, 32768, 131072]);
  });

  it('caps every column at maxPositionEmbeddings, dropping table contexts beyond it', () => {
    const model = makeSpec({ maxPositionEmbeddings: 4096 });
    expect(fitMatrixContexts(model, 4096)).toEqual([2048, 4096]);
  });

  it('never emits a column past maxPositionEmbeddings even when the chosen context is smaller', () => {
    const model = makeSpec({ maxPositionEmbeddings: 4096 });
    expect(fitMatrixContexts(model, 2048)).toEqual([2048, 4096]);
  });
});

describe('fitMatrix', () => {
  it('covers every weight quant by default, in WEIGHT_QUANTS order', () => {
    const m = fitMatrix(state());
    expect(m.rows.map((r) => r.weight)).toEqual(Object.keys(WEIGHT_QUANTS));
    expect(FIT_MATRIX_QUANTS).toEqual(Object.keys(WEIGHT_QUANTS));
  });

  it('each cell equals maxUsersAtContext computed one cell at a time via calculate()', () => {
    const s = state();
    const m = fitMatrix(s);
    for (const row of m.rows) {
      for (const cell of row.cells) {
        const oneOff = calculate({
          ...s,
          quant: { ...s.quant, weight: row.weight },
          workload: { ...s.workload, contextTokens: cell.contextTokens },
        });
        expect(cell.maxUsers).toBe(oneOff.maxUsersAtContext);
        expect(cell.fits).toBe(oneOff.fits);
        expect(cell.headroomBytes).toBe(oneOff.headroomBytes);
        expect(cell.usableBytes).toBe(oneOff.usableBytes);
      }
    }
  });

  it('respects explicit quants and contexts arguments', () => {
    const m = fitMatrix(state(), ['fp8', 'q4_k_m'], [2048, 8192]);
    expect(m.contexts).toEqual([2048, 8192]);
    expect(m.rows.map((r) => r.weight)).toEqual(['fp8', 'q4_k_m']);
    expect(m.rows[0].cells.map((c) => c.contextTokens)).toEqual([2048, 8192]);
  });

  it('a smaller weight quant fits more users at the same context than a larger one', () => {
    const bigHw: HardwareSpec = { ...hw, gpuCount: 4, vramGB: 80 };
    const m = fitMatrix(state({ hardware: bigHw }), ['fp32', 'q4_k_m'], [8192]);
    const fp32Users = m.rows[0].cells[0].maxUsers;
    const q4Users = m.rows[1].cells[0].maxUsers;
    expect(q4Users).toBeGreaterThan(fp32Users);
  });

  it('a row whose quant does not match loaded file weights falls back to the estimate automatically', () => {
    const model = makeSpec({
      params: 7e9,
      activeParams: 7e9,
      nativeDtype: 'bf16',
      fileWeights: { bytes: 4e9, label: 'Q4_K_M GGUF', quant: 'q4_k_m' },
    });
    const s = state({ model, quant: { weight: 'q4_k_m', kv: 'fp16' } });
    const m = fitMatrix(s, ['q4_k_m', 'fp16'], [8192]);
    const q4kmResult = calculate({ ...s, quant: { weight: 'q4_k_m', kv: 'fp16' }, workload: { ...s.workload, contextTokens: 8192 } });
    const fp16Result = calculate({ ...s, quant: { weight: 'fp16', kv: 'fp16' }, workload: { ...s.workload, contextTokens: 8192 } });
    expect(m.rows[0].cells[0].maxUsers).toBe(q4kmResult.maxUsersAtContext);
    expect(m.rows[1].cells[0].maxUsers).toBe(fp16Result.maxUsersAtContext);
    // The fp16 row must NOT reuse the file bytes pinned to q4_k_m: it should be the (larger) estimate.
    expect(fp16Result.weightSource).toBe('estimate');
    expect(fp16Result.weightBytes).toBeGreaterThan(q4kmResult.weightBytes);
  });
});
