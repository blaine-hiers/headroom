import { fitMatrix, formatNumber, formatTokens, WEIGHT_QUANTS } from '../lib';
import type { CalcState, WeightQuantKey } from '../lib';
import { fitLevel } from './state';

interface Props {
  state: CalcState;
  onApply: (weight: WeightQuantKey, contextTokens: number) => void;
}

const cellUsers = (n: number) => (Number.isFinite(n) ? formatNumber(n) : '∞');

/**
 * Weight quant x context length heatmap: click a cell to load that quant and context into the
 * calculator. Colour comes from `fitLevel`, the same fits/tight/nofit classifier the verdict
 * badge uses, applied to the CURRENTLY configured user count at that cell's quant + context. The
 * number in the cell is that cell's own max users (independent of the configured user count) —
 * see fitMatrix.ts.
 */
export function FitMatrix({ state, onApply }: Props) {
  const matrix = fitMatrix(state);
  const { quant, workload } = state;
  const chosenContext = Math.floor(workload.contextTokens);

  return (
    <details className="card disclosure fit-matrix">
      <summary>Fit matrix: weight quant × context</summary>
      <p className="help">
        Max concurrent users at each weight quant and context, for {state.hardware.gpuCount} × {state.hardware.gpuName}. Colour shows whether{' '}
        {formatNumber(workload.concurrentUsers)} user{workload.concurrentUsers === 1 ? '' : 's'} fits. Click a cell to apply it.
      </p>
      <div className="table-wrap">
        <table className="fit-matrix-table">
          <thead>
            <tr>
              <th scope="col">Weight quant</th>
              {matrix.contexts.map((c) => (
                <th key={c} scope="col" className="num">
                  {formatTokens(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.rows.map((row) => (
              <tr key={row.weight}>
                <th scope="row">{WEIGHT_QUANTS[row.weight].label}</th>
                {row.cells.map((cell) => {
                  const level = fitLevel(cell.fits, cell.headroomBytes, cell.usableBytes);
                  const chosen = row.weight === quant.weight && cell.contextTokens === chosenContext;
                  const label = `${WEIGHT_QUANTS[row.weight].label} at ${formatTokens(cell.contextTokens)}: ${
                    cell.maxUsers > 0 ? `${cellUsers(cell.maxUsers)} max users` : 'does not fit'
                  }`;
                  return (
                    <td key={cell.contextTokens} className="fit-matrix-cell-wrap">
                      <button
                        type="button"
                        className={`fit-cell fit-cell-${level}`}
                        aria-pressed={chosen}
                        aria-label={label}
                        title={label}
                        onClick={() => onApply(row.weight, cell.contextTokens)}
                      >
                        {cell.maxUsers > 0 ? cellUsers(cell.maxUsers) : '✕'}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
