import { describe, expect, it } from 'vitest';
import { contextTableToCsv, csvField, fitMatrixToCsv } from './exportCsv';
import type { FitMatrix } from './fitMatrix';

describe('csvField', () => {
  it('leaves plain fields unescaped', () => {
    expect(csvField('hello')).toBe('hello');
    expect(csvField(42)).toBe('42');
  });

  it('quotes and escapes a field containing a comma, quote or newline', () => {
    expect(csvField('a,b')).toBe('"a,b"');
    expect(csvField('a"b')).toBe('"a""b"');
    expect(csvField('a\nb')).toBe('"a\nb"');
    expect(csvField('a\r\nb')).toBe('"a\r\nb"');
  });
});

describe('contextTableToCsv', () => {
  it('renders a header row and one row per context', () => {
    const csv = contextTableToCsv([
      { contextTokens: 2048, kvBytesPerRequest: 1000, maxUsers: 10 },
      { contextTokens: 8192, kvBytesPerRequest: 4000, maxUsers: 0 },
    ]);
    const lines = csv.split('\r\n');
    expect(lines[0]).toBe('Context tokens,KV per request (bytes),Max users');
    expect(lines[1]).toBe('2048,1000,10');
    expect(lines[2]).toBe('8192,4000,0');
    // trailing CRLF, no dangling blank content after it
    expect(lines[3]).toBe('');
  });

  it('spells out an unbounded max-users figure rather than emitting Infinity', () => {
    const csv = contextTableToCsv([{ contextTokens: 0, kvBytesPerRequest: 0, maxUsers: Number.POSITIVE_INFINITY }]);
    expect(csv).toContain('0,0,unbounded');
  });
});

describe('fitMatrixToCsv', () => {
  it('renders one row per weight quant with a max-users column per context', () => {
    const matrix: FitMatrix = {
      contexts: [2048, 8192],
      rows: [
        {
          weight: 'q4_k_m',
          cells: [
            { weight: 'q4_k_m', contextTokens: 2048, maxUsers: 12, fits: true, headroomBytes: 1, usableBytes: 2 },
            { weight: 'q4_k_m', contextTokens: 8192, maxUsers: 0, fits: false, headroomBytes: -1, usableBytes: 2 },
          ],
        },
        {
          weight: 'fp16',
          cells: [
            { weight: 'fp16', contextTokens: 2048, maxUsers: 4, fits: true, headroomBytes: 1, usableBytes: 2 },
            { weight: 'fp16', contextTokens: 8192, maxUsers: 0, fits: false, headroomBytes: -1, usableBytes: 2 },
          ],
        },
      ],
    };
    const lines = fitMatrixToCsv(matrix).split('\r\n');
    expect(lines[0]).toBe('Weight quant,2048 tokens,8192 tokens');
    expect(lines[1]).toBe('Q4_K_M,12,0');
    expect(lines[2]).toBe('FP16,4,0');
  });
});
