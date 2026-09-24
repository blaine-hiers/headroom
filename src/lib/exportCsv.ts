import type { FitMatrix } from './fitMatrix';
import { WEIGHT_QUANTS } from './quant';
import type { CalcResult } from './types';

/** RFC 4180 line ending; Excel and most spreadsheet tools expect CRLF. */
const CRLF = '\r\n';

/** Escapes one CSV field: wraps it in quotes when it contains a comma, quote or newline, doubling any inner quotes. */
export function csvField(value: string | number): string {
  const s = String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function csvRow(fields: Array<string | number>): string {
  return fields.map(csvField).join(',');
}

/** Unbounded max-users (no per-token KV cost, e.g. a 0-token context) reads as "∞" on the page; CSV spells it out. */
const usersCell = (n: number): string | number => (Number.isFinite(n) ? n : 'unbounded');

/** The Results "Context table" card as CSV: one row per context length. */
export function contextTableToCsv(rows: CalcResult['contextTable']): string {
  const lines = [csvRow(['Context tokens', 'KV per request (bytes)', 'Max users'])];
  for (const row of rows) {
    lines.push(csvRow([row.contextTokens, row.kvBytesPerRequest, usersCell(row.maxUsers)]));
  }
  return lines.join(CRLF) + CRLF;
}

/** The fit matrix (weight quant × context) as CSV: one row per weight quant, one column per context, max users per cell. */
export function fitMatrixToCsv(matrix: FitMatrix): string {
  const header = ['Weight quant', ...matrix.contexts.map((c) => `${c} tokens`)];
  const lines = [csvRow(header)];
  for (const row of matrix.rows) {
    lines.push(csvRow([WEIGHT_QUANTS[row.weight].label, ...row.cells.map((cell) => usersCell(cell.maxUsers))]));
  }
  return lines.join(CRLF) + CRLF;
}
