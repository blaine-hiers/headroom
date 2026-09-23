import { useId, useState } from 'react';
import type { ReactNode } from 'react';
import { clamp } from './state';

interface Props {
  label: ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  /** Decimal places shown when not being edited. */
  decimals?: number;
  /** Round the committed value to an integer. */
  integer?: boolean;
  suffix?: string;
  help?: ReactNode;
  className?: string;
  /**
   * Commit only on blur or Enter, not per keystroke. For fields whose partial values would
   * clamp other state (Max position caps the workload context).
   */
  commitOnBlur?: boolean;
}

function show(v: number, decimals: number | undefined): string {
  if (!Number.isFinite(v)) return '';
  return decimals === undefined ? String(v) : v.toFixed(decimals);
}

/**
 * Numeric input that never emits NaN or an out-of-range value. While focused the
 * raw text is kept so partial input ("1.", "") can be typed; on blur it snaps back
 * to the clamped value.
 */
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  decimals,
  integer = false,
  suffix,
  help,
  className,
  commitOnBlur = false,
}: Props) {
  const id = useId();
  const helpId = `${id}-help`;
  const [draft, setDraft] = useState<string | null>(null);
  const edited = draft !== null && draft !== show(value, decimals);

  // While typing, a value below `min` is usually a prefix of a bigger number ("4" on the way
  // to "4096"); committing it clamped would push dependent state (e.g. the workload context)
  // down to the minimum for good. Hold it until blur, then clamp.
  const commit = (raw: string, final = false) => {
    if (raw.trim() === '') return;
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    if (!final && n < min) return;
    let v = clamp(n, min, max);
    if (integer) v = Math.round(v);
    onChange(v);
  };

  return (
    <div className={className ? `field ${className}` : 'field'}>
      <label htmlFor={id}>{label}</label>
      <div className="input-wrap">
        <input
          id={id}
          type="number"
          inputMode={integer ? 'numeric' : 'decimal'}
          min={min}
          max={max}
          step={step}
          value={draft ?? show(value, decimals)}
          aria-describedby={help ? helpId : undefined}
          onFocus={() => setDraft(show(value, decimals))}
          onChange={(e) => {
            setDraft(e.target.value);
            if (!commitOnBlur) commit(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && edited) commit(draft, true);
          }}
          onBlur={() => {
            // Only an edited draft commits: tabbing through must not round a value or mark it manual.
            if (edited) commit(draft, true);
            setDraft(null);
          }}
        />
        {suffix && <span className="suffix">{suffix}</span>}
      </div>
      {help && (
        <p className="help" id={helpId}>
          {help}
        </p>
      )}
    </div>
  );
}

interface StepperProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  help?: ReactNode;
}

/** Integer input with − / + buttons. */
export function Stepper({ label, value, onChange, min, max, help }: StepperProps) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const set = (v: number) => onChange(Math.round(clamp(v, min, max)));
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="stepper">
        <button type="button" aria-label={`Decrease ${label.toLowerCase()}`} disabled={value <= min} onClick={() => set(value - 1)}>
          −
        </button>
        <input
          id={id}
          type="number"
          inputMode="numeric"
          min={min}
          max={max}
          step={1}
          value={draft ?? String(value)}
          onFocus={() => setDraft(String(value))}
          onChange={(e) => {
            setDraft(e.target.value);
            const n = Number(e.target.value);
            if (e.target.value.trim() !== '' && Number.isFinite(n)) set(n);
          }}
          onBlur={() => setDraft(null)}
        />
        <button type="button" aria-label={`Increase ${label.toLowerCase()}`} disabled={value >= max} onClick={() => set(value + 1)}>
          +
        </button>
      </div>
      {help && <p className="help">{help}</p>}
    </div>
  );
}
