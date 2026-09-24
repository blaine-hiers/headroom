import type { ReactNode } from 'react';
import { bestColumnIndex, columnLabel, COMPARE_ROWS, formatNumber, formatTokens } from '../lib';
import type { CalcResult } from '../lib';
import { Bytes } from './Bytes';

const MAX_COMPARE_COLUMNS = 3;

export interface CompareColumn {
  result: CalcResult;
  /** Short summary shown under the column letter, e.g. "Llama 3.3 70B · RTX 4090 ×1". */
  summary: string;
}

interface Props {
  columns: CompareColumn[];
  selected: number;
  onSelect: (index: number) => void;
  onDuplicate: () => void;
  onRemove: (index: number) => void;
}

function tokS(v: number): string {
  return v > 0 && Number.isFinite(v) ? formatNumber(v, v < 10 ? 1 : 0) : '—';
}

function cellText(unit: 'bytes' | 'count' | 'tokens' | 'tokS', value: number): ReactNode {
  switch (unit) {
    case 'bytes':
      return <Bytes value={value} />;
    case 'tokens':
      return formatTokens(value);
    case 'tokS':
      return tokS(value);
    default:
      return formatNumber(value);
  }
}

/** Compare mode's headline table: one column per config, the better value per row highlighted. */
export function CompareTable({ columns, selected, onSelect, onDuplicate, onRemove }: Props) {
  return (
    <div className="card compare-table" aria-labelledby="compare-h">
      <div className="compare-table-head">
        <h3 id="compare-h">Compare</h3>
        {columns.length < MAX_COMPARE_COLUMNS && (
          <button type="button" className="btn btn-ghost" onClick={onDuplicate}>
            Duplicate current config
          </button>
        )}
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Config</th>
              {columns.map((c, i) => (
                <th key={i} scope="col" aria-current={i === selected ? 'true' : undefined}>
                  <div className="compare-col-head">
                    <button type="button" className={`btn btn-ghost compare-select${i === selected ? ' compare-selected' : ''}`} onClick={() => onSelect(i)}>
                      {columnLabel(i)}
                    </button>
                    {i > 0 && (
                      <button type="button" className="link-btn compare-remove" onClick={() => onRemove(i)} aria-label={`Remove config ${columnLabel(i)}`}>
                        Remove
                      </button>
                    )}
                  </div>
                  <p className="muted compare-summary">{c.summary}</p>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPARE_ROWS.map((row) => {
              const values = columns.map((c) => row.value(c.result));
              const winner = bestColumnIndex(values, row.higherIsBetter);
              return (
                <tr key={row.key}>
                  <th scope="row">{row.label}</th>
                  {values.map((v, i) => (
                    <td key={i} className={i === winner ? 'compare-best' : undefined}>
                      {cellText(row.unit, v)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
