import { calculate, TABLE_CONTEXTS } from './fit';
import { WEIGHT_QUANTS } from './quant';
import type { CalcState, ModelSpec, WeightQuantKey } from './types';

/** Every weight quant, in the table's display order (same as WEIGHT_QUANTS). */
export const FIT_MATRIX_QUANTS = Object.keys(WEIGHT_QUANTS) as WeightQuantKey[];

/**
 * 2K, 8K, 32K, 128K (only the ones the model can reach) plus maxPositionEmbeddings and the
 * chosen context, ascending, deduplicated. Columns are never shown past the model's own max —
 * there is nothing to compute there.
 */
export function fitMatrixContexts(model: ModelSpec, chosenContext: number): number[] {
  const maxPos = model.maxPositionEmbeddings;
  const cols = new Set<number>();
  for (const c of TABLE_CONTEXTS) {
    if (c <= maxPos) cols.add(c);
  }
  if (maxPos > 0) cols.add(maxPos);
  if (chosenContext > 0 && chosenContext <= maxPos) cols.add(chosenContext);
  return [...cols].sort((a, b) => a - b);
}

export interface FitMatrixCell {
  weight: WeightQuantKey;
  contextTokens: number;
  /** Max concurrent users at this quant + context (independent of the configured user count). */
  maxUsers: number;
  /**
   * Whether the currently configured user count runs at this quant + context (CalcResult.runs:
   * fits in VRAM, or runs with CPU/RAM offload when that's on).
   */
  runs: boolean;
  /** CalcResult.runHeadroomBytes — the headroom that matches `runs`. */
  headroomBytes: number;
  usableBytes: number;
}

export interface FitMatrixRow {
  weight: WeightQuantKey;
  cells: FitMatrixCell[];
}

export interface FitMatrix {
  contexts: number[];
  rows: FitMatrixRow[];
}

/**
 * Weight quant (rows) x context length (columns) fit matrix for `state`'s current model, GPUs
 * and user count. Each cell re-runs `calculate()` with that row's weight quant and that column's
 * context swapped in, so every other feature's rules (runtime profiles, tensor-parallel KV
 * replication, GGUF/file-weight fallback, prefill, ...) apply automatically — including
 * `resolveWeights` falling back to the estimate for any row whose quant isn't the one the
 * loaded files are in.
 */
export function fitMatrix(
  state: CalcState,
  quants: WeightQuantKey[] = FIT_MATRIX_QUANTS,
  contexts: number[] = fitMatrixContexts(state.model, state.workload.contextTokens),
): FitMatrix {
  const rows = quants.map((weight) => ({
    weight,
    cells: contexts.map((contextTokens) => {
      const result = calculate({
        ...state,
        quant: { ...state.quant, weight },
        workload: { ...state.workload, contextTokens },
      });
      return {
        weight,
        contextTokens,
        maxUsers: result.maxUsersAtContext,
        runs: result.runs,
        headroomBytes: result.runHeadroomBytes,
        usableBytes: result.usableBytes,
      };
    }),
  }));
  return { contexts, rows };
}
