const DASH = '—';
const DECIMAL_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
const BINARY_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];

/** ~3 significant figures, trailing zeros dropped: 2.68, 2.5, 10.7, 40, 671. */
function sig3(v: number): string {
  const abs = Math.abs(v);
  const decimals = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return String(Number(v.toFixed(decimals)));
}

function scale(bytes: number, base: number, units: string[]): string {
  if (!Number.isFinite(bytes)) return DASH;
  const sign = bytes < 0 ? '-' : '';
  let v = Math.abs(bytes);
  let i = 0;
  // Promote while the rounded value would read as >= one of the next unit ("1000 KB" → "1 MB").
  while (i < units.length - 1 && Number(sig3(v)) >= base) {
    v /= base;
    i++;
  }
  return `${sign}${sig3(v)} ${units[i]}`;
}

/** Decimal (1000-based) units, matching vendor VRAM figures: "328 KB", "2.5 GB", "1.2 TB". */
export function formatBytes(bytes: number): string {
  return scale(bytes, 1000, DECIMAL_UNITS);
}

/** Binary (1024-based) units for tooltips: "320 KiB", "640 MiB", "40 GiB". */
export function formatBytesBinary(bytes: number): string {
  return scale(bytes, 1024, BINARY_UNITS);
}

export function formatNumber(n: number, maxFractionDigits = 0): string {
  if (!Number.isFinite(n)) return DASH;
  return n.toLocaleString('en-US', { maximumFractionDigits: maxFractionDigits });
}

/** Token counts in 1024s: 2048 → "2K", 131072 → "128K", 1048576 → "1M". */
export function formatTokens(n: number): string {
  if (!Number.isFinite(n)) return DASH;
  const abs = Math.abs(n);
  if (abs >= 1024 * 1024) return `${Number((n / (1024 * 1024)).toFixed(1))}M`;
  if (abs >= 1024) return `${Number((n / 1024).toFixed(1))}K`;
  return String(Math.round(n));
}

/** USD amounts: "$6.52", "$104.32", "$0.0042" (finer precision below a cent so small $/1M-token figures don't round to $0.00). */
export function formatUsd(n: number): string {
  if (!Number.isFinite(n)) return DASH;
  const decimals = n !== 0 && Math.abs(n) < 0.01 ? 4 : 2;
  return `$${n.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

/** Seconds → "120 ms", "1.2 s", "3.4 min" for TTFT-scale durations. */
export function formatSeconds(s: number): string {
  if (!Number.isFinite(s) || s < 0) return DASH;
  if (s < 1) return `${sig3(s * 1000)} ms`;
  if (s < 60) return `${sig3(s)} s`;
  return `${sig3(s / 60)} min`;
}
