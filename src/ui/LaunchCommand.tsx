import { useEffect, useRef, useState } from 'react';
import { buildLaunchCommand } from '../lib';
import type { CalcState } from '../lib';

interface Props {
  state: CalcState;
}

/** Ready-to-paste launch command for the chosen runtime, built from the current calculator state. */
export function LaunchCommand({ state }: Props) {
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle');
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const { command, notes } = buildLaunchCommand(state);
  if (command === null) return null;

  const copy = async () => {
    let ok = false;
    try {
      await navigator.clipboard.writeText(command);
      ok = true;
    } catch {
      ok = false;
    }
    setCopied(ok ? 'copied' : 'failed');
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied('idle'), 1500);
  };

  return (
    <div className="card launch-command">
      <h3>Launch command</h3>
      <div className="code-block">
        <pre>
          <code>{command}</code>
        </pre>
        <button type="button" className="btn btn-ghost" onClick={() => void copy()}>
          <span aria-live="polite">{copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Copy failed' : 'Copy'}</span>
        </button>
      </div>
      {notes.map((note) => (
        <p className="help" key={note}>
          {note}
        </p>
      ))}
    </div>
  );
}
