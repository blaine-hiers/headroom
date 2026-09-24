import { useId } from 'react';
import { DEFAULT_MODEL_PRESET, MODEL_PRESETS, WEIGHT_QUANTS } from '../lib';
import type { DraftMode, SpeculativeConfig, WeightQuantKey } from '../lib';
import { NumberField } from './NumberField';
import { MAX_DRAFT_K } from './state';

interface Props {
  speculative: SpeculativeConfig;
  onChange: (patch: Partial<SpeculativeConfig>) => void;
}

const BIG_B = 1000; // 1000B = 1T params, a generous manual-entry ceiling

export function SpeculativeFields({ speculative, onChange }: Props) {
  const id = useId();
  const mode = speculative.draftMode;

  const setMode = (next: DraftMode) => {
    if (next === 'preset') {
      onChange({ draftMode: 'preset', draftModel: speculative.draftModel ?? DEFAULT_MODEL_PRESET, draftParams: undefined });
    } else if (next === 'custom') {
      onChange({ draftMode: 'custom', draftModel: undefined, draftParams: speculative.draftParams ?? 1e9 });
    } else {
      onChange({ draftMode: 'none', draftModel: undefined, draftParams: undefined });
    }
  };

  return (
    <details className="card collapsible">
      <summary>Speculative decoding</summary>
      <div className="collapsible-body">
        <label className="check">
          <input type="checkbox" checked={speculative.enabled} onChange={(e) => onChange({ enabled: e.target.checked })} />
          Enable
        </label>

        {speculative.enabled && (
          <>
            <div className="field">
              <label htmlFor={`${id}-mode`}>Draft model</label>
              <select id={`${id}-mode`} value={mode} onChange={(e) => setMode(e.target.value as DraftMode)}>
                <option value="none">None — n-gram / prompt lookup</option>
                <option value="preset">Preset model</option>
                <option value="custom">Custom (enter params)</option>
              </select>
              {mode === 'none' && <p className="help">No extra memory or per-step cost, but acceptance is usually lower than a trained draft model.</p>}
            </div>

            {mode === 'preset' && (
              <div className="field">
                <label htmlFor={`${id}-preset`}>Preset</label>
                <select
                  id={`${id}-preset`}
                  value={speculative.draftModel?.id ?? DEFAULT_MODEL_PRESET.id}
                  onChange={(e) => {
                    const spec = MODEL_PRESETS.find((p) => p.id === e.target.value);
                    if (spec) onChange({ draftModel: spec });
                  }}
                >
                  {MODEL_PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <p className="help">Its weights and its own KV cache (at the same context and user count) are added to the fit.</p>
              </div>
            )}

            {mode === 'custom' && (
              <NumberField
                label="Draft parameters"
                suffix="B"
                value={(speculative.draftParams ?? 0) / 1e9}
                decimals={2}
                step={0.01}
                min={0}
                max={BIG_B}
                onChange={(v) => onChange({ draftParams: Math.round(v * 1e9) })}
                help="Weights only — KV cache isn't estimated without a known architecture; pick a preset for that."
              />
            )}

            {mode !== 'none' && (
              <div className="field">
                <label htmlFor={`${id}-quant`}>Draft weight quant</label>
                <select id={`${id}-quant`} value={speculative.draftWeightQuant} onChange={(e) => onChange({ draftWeightQuant: e.target.value as WeightQuantKey })}>
                  {(Object.keys(WEIGHT_QUANTS) as WeightQuantKey[]).map((k) => (
                    <option key={k} value={k}>
                      {WEIGHT_QUANTS[k].label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="grid2">
              <NumberField
                label="Draft tokens per step (k)"
                value={speculative.k}
                integer
                min={0}
                max={MAX_DRAFT_K}
                onChange={(v) => onChange({ k: v })}
              />
              <NumberField
                label="Acceptance rate (α)"
                value={speculative.alpha}
                decimals={2}
                step={0.01}
                min={0}
                max={1}
                onChange={(v) => onChange({ alpha: v })}
                help="Workload-dependent — measure it for your own prompts; this is only a starting guess."
              />
            </div>
          </>
        )}
      </div>
    </details>
  );
}
