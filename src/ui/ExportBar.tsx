import { useEffect, useRef, useState } from 'react';
import { buildMarkdownSummary, contextTableToCsv, fitMatrix, fitMatrixToCsv } from '../lib';
import type { CalcResult, CalcState } from '../lib';

interface Props {
  state: CalcState;
  result: CalcResult;
  /** Writes the current state into the URL and returns the full shareable link (same one Header's Copy link uses). */
  getLink: () => string;
}

/** Triggers a browser download of `content` as a file, entirely client-side (a Blob object URL, no server). */
function downloadCsv(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Copy-as-Markdown and download-CSV actions for sharing a result outside the page. */
export function ExportBar({ state, result, getLink }: Props) {
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle');
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copyMarkdown = async () => {
    const markdown = buildMarkdownSummary(state, result, getLink());
    let ok = false;
    try {
      await navigator.clipboard.writeText(markdown);
      ok = true;
    } catch {
      ok = false;
    }
    setCopied(ok ? 'copied' : 'failed');
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied('idle'), 1500);
  };

  return (
    <div className="card export-bar">
      <h3>Export</h3>
      <nav className="header-actions" aria-label="Export actions">
        <button type="button" className="btn btn-ghost" onClick={() => void copyMarkdown()}>
          <span aria-live="polite">{copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Copy failed' : 'Copy as Markdown'}</span>
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => downloadCsv('headroom-context-table.csv', contextTableToCsv(result.contextTable))}>
          Download context table (CSV)
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => downloadCsv('headroom-fit-matrix.csv', fitMatrixToCsv(fitMatrix(state)))}>
          Download fit matrix (CSV)
        </button>
      </nav>
      <p className="help">Markdown includes the headline figures, the context table and a link back to this configuration. CSV downloads happen entirely in your browser.</p>
    </div>
  );
}
