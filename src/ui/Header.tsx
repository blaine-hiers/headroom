import { useEffect, useRef, useState } from 'react';
import { ThemeToggle } from './ThemeToggle';

interface Props {
  /** Writes the state into the URL and returns the full shareable link. */
  getLink: () => string;
  compareOn: boolean;
  onToggleCompare: () => void;
}

export function Header({ getLink, compareOn, onToggleCompare }: Props) {
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle');
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copy = async () => {
    const link = getLink();
    let ok = false;
    try {
      await navigator.clipboard.writeText(link);
      ok = true;
    } catch {
      ok = false;
    }
    setCopied(ok ? 'copied' : 'failed');
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied('idle'), 1500);
  };

  return (
    <header className="header">
      <div className="brand">
        <svg className="logo" viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
          <rect x="3" y="3" width="26" height="26" rx="6" fill="var(--surface-2)" stroke="var(--border-strong)" />
          <rect x="8" y="17" width="4" height="8" rx="1" fill="var(--ok)" />
          <rect x="14" y="12" width="4" height="13" rx="1" fill="var(--ok)" />
          <rect x="20" y="7" width="4" height="18" rx="1" fill="var(--accent)" />
        </svg>
        <div>
          <h1>Headroom</h1>
          <p className="tagline">Will it fit? For how many? How fast?</p>
        </div>
      </div>
      <nav className="header-actions" aria-label="Page actions">
        <button type="button" className="btn btn-ghost" aria-pressed={compareOn} onClick={onToggleCompare}>
          {compareOn ? 'Compare: on' : 'Compare'}
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => void copy()}>
          <span aria-live="polite">{copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Link is in the address bar' : 'Copy link'}</span>
        </button>
        <ThemeToggle />
      </nav>
    </header>
  );
}
