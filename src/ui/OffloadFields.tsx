import { useId } from 'react';
import { RAM_PRESETS, resolveOffload } from '../lib';
import type { HardwareSpec, OffloadSpec } from '../lib';
import { NumberField } from './NumberField';

interface Props {
  hardware: HardwareSpec;
  onChange: (patch: Partial<HardwareSpec>) => void;
}

/** CPU/RAM layer offload inputs (llama.cpp's -ngl). Off by default; see src/lib/offload.ts. */
export function OffloadFields({ hardware, onChange }: Props) {
  const offload = resolveOffload(hardware.offload);
  const presetId = useId();
  const setOffload = (patch: Partial<OffloadSpec>) => onChange({ offload: { ...offload, ...patch } });

  return (
    <details className="panel-details">
      <summary>Offload (CPU / system RAM)</summary>
      <p>
        Layers that don't fit in VRAM run from system RAM instead, like llama.cpp's <code>-ngl</code>. Off by default —
        today's fit and throughput numbers are unchanged until you turn this on. KV cache always stays on the GPU.
      </p>
      <label className="check">
        <input type="checkbox" checked={offload.enabled} onChange={(e) => setOffload({ enabled: e.target.checked })} />
        Offload layers that don't fit to system RAM
      </label>
      {offload.enabled && (
        <>
          <div className="grid2">
            <NumberField
              label="System RAM"
              suffix="GB"
              value={offload.systemRamGB}
              min={0.1}
              max={8192}
              step={1}
              onChange={(v) => setOffload({ systemRamGB: v })}
            />
            <NumberField
              label="RAM bandwidth"
              suffix="GB/s"
              value={offload.ramBandwidthGBs}
              min={1}
              max={2000}
              step={1}
              onChange={(v) => setOffload({ ramBandwidthGBs: v })}
            />
          </div>
          <div className="field">
            <label htmlFor={presetId}>RAM bandwidth preset</label>
            <select
              id={presetId}
              value=""
              onChange={(e) => {
                const preset = RAM_PRESETS.find((p) => p.label === e.target.value);
                if (preset && preset.bandwidthGBs !== null) setOffload({ ramBandwidthGBs: preset.bandwidthGBs });
                e.target.value = '';
              }}
            >
              <option value="" disabled>
                Choose a preset…
              </option>
              {RAM_PRESETS.map((p) => (
                <option key={p.label} value={p.label} disabled={p.bandwidthGBs === null}>
                  {p.label}
                </option>
              ))}
            </select>
            <small>Apple has no separate system RAM — it's unified with the GPU, so use the GPU bandwidth above.</small>
          </div>
        </>
      )}
    </details>
  );
}
