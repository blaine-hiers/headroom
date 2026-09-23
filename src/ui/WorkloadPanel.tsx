import { useId } from 'react';
import { formatTokens } from '../lib';
import type { Workload } from '../lib';
import { NumberField, Stepper } from './NumberField';
import { MAX_USERS, MIN_CONTEXT } from './state';

interface Props {
  workload: Workload;
  maxContext: number;
  onChange: (patch: Partial<Workload>) => void;
}

const MIN_EXP = Math.log2(MIN_CONTEXT);

export function WorkloadPanel({ workload, maxContext, onChange }: Props) {
  const sliderId = useId();
  // Log-scale slider over exponents; it snaps to powers of two and the top stop clamps to the model max.
  const maxExp = Math.max(MIN_EXP, Math.ceil(Math.log2(maxContext)));
  // At the model max (often not a power of two, e.g. 40960) the thumb sits on the top stop,
  // so ArrowLeft steps to the power of two just below it instead of skipping one.
  const exp =
    workload.contextTokens >= maxContext ? maxExp : Math.round(Math.log2(Math.max(MIN_CONTEXT, workload.contextTokens)));
  return (
    <section className="panel" aria-labelledby="wl-h">
      <h2 id="wl-h">Workload</h2>
      <div className="field">
        <label htmlFor={sliderId}>
          Context length <span className="muted num">{formatTokens(workload.contextTokens)} tokens</span>
        </label>
        <input
          id={sliderId}
          type="range"
          min={MIN_EXP}
          max={maxExp}
          step={1}
          value={exp}
          aria-valuetext={`${workload.contextTokens} tokens`}
          onChange={(e) => onChange({ contextTokens: Math.min(maxContext, 2 ** Number(e.target.value)) })}
        />
        <div className="range-scale muted num" aria-hidden="true">
          <span>{formatTokens(MIN_CONTEXT)}</span>
          <span>{formatTokens(maxContext)}</span>
        </div>
      </div>
      <NumberField
        label="Context tokens"
        suffix="tok"
        value={workload.contextTokens}
        integer
        min={MIN_CONTEXT}
        max={maxContext}
        step={256}
        onChange={(v) => onChange({ contextTokens: v })}
        help={`Per request; the model supports up to ${maxContext.toLocaleString('en-US')}.`}
      />
      <Stepper
        label="Concurrent users"
        value={workload.concurrentUsers}
        min={1}
        max={MAX_USERS}
        onChange={(v) => onChange({ concurrentUsers: v })}
        help="Requests decoding at the same time, each holding a full context."
      />
    </section>
  );
}
