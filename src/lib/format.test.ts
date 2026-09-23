import { describe, expect, it } from 'vitest';
import { formatBytes, formatBytesBinary, formatNumber, formatTokens } from './format';

describe('formatBytes (decimal)', () => {
  it('uses sensible precision', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(320_000)).toBe('320 KB');
    expect(formatBytes(2.5e9)).toBe('2.5 GB');
    expect(formatBytes(40e9)).toBe('40 GB');
    expect(formatBytes(1.2e12)).toBe('1.2 TB');
    expect(formatBytes(2_684_354_560)).toBe('2.68 GB');
  });
  it('promotes values that round up to the next unit', () => {
    expect(formatBytes(999_999)).toBe('1 MB');
    expect(formatBytes(999)).toBe('999 B');
  });
  it('keeps the sign for negative headroom', () => {
    expect(formatBytes(-3.2e9)).toBe('-3.2 GB');
  });
  it('renders non-finite as a dash', () => {
    expect(formatBytes(NaN)).toBe('—');
    expect(formatBytes(Infinity)).toBe('—');
  });
});

describe('formatBytesBinary', () => {
  it('uses 1024-based units', () => {
    expect(formatBytesBinary(671_088_640)).toBe('640 MiB');
    expect(formatBytesBinary(1024)).toBe('1 KiB');
    expect(formatBytesBinary(40 * 1024 ** 3)).toBe('40 GiB');
    expect(formatBytesBinary(NaN)).toBe('—');
  });
});

describe('formatNumber', () => {
  it('groups thousands and limits decimals', () => {
    expect(formatNumber(1234567)).toBe('1,234,567');
    expect(formatNumber(43.912, 1)).toBe('43.9');
    expect(formatNumber(Infinity)).toBe('—');
  });
});

describe('formatTokens', () => {
  it('uses 1024-based K/M', () => {
    expect(formatTokens(2048)).toBe('2K');
    expect(formatTokens(131072)).toBe('128K');
    expect(formatTokens(40960)).toBe('40K');
    expect(formatTokens(1536)).toBe('1.5K');
    expect(formatTokens(1_048_576)).toBe('1M');
    expect(formatTokens(256)).toBe('256');
    expect(formatTokens(NaN)).toBe('—');
  });
});
