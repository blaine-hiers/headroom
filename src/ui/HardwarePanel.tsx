import { useId } from 'react';
import { GPU_PRESETS, findGpuPreset } from '../lib';
import type { GpuVendor, HardwareSpec } from '../lib';
import { NumberField, Stepper } from './NumberField';
import { OffloadFields } from './OffloadFields';
import { MAX_GPUS } from './state';

const VENDORS: Array<{ vendor: GpuVendor; label: string }> = [
  { vendor: 'nvidia-consumer', label: 'NVIDIA consumer / workstation' },
  { vendor: 'nvidia-datacenter', label: 'NVIDIA datacenter' },
  { vendor: 'amd', label: 'AMD' },
  { vendor: 'apple', label: 'Apple (unified memory)' },
  { vendor: 'other', label: 'Other' },
];

interface Props {
  hardware: HardwareSpec;
  onChange: (patch: Partial<HardwareSpec>) => void;
}

export function HardwarePanel({ hardware, onChange }: Props) {
  const gpuId = useId();
  return (
    <section className="panel" aria-labelledby="hw-h">
      <h2 id="hw-h">Hardware</h2>
      <div className="field">
        <label htmlFor={gpuId}>GPU</label>
        <select
          id={gpuId}
          value={hardware.gpuName}
          onChange={(e) => {
            const newGpuName = e.target.value;
            const newGpu = findGpuPreset(newGpuName);
            const patch: Partial<HardwareSpec> = { gpuName: newGpuName };
            // Clear appleWiredLimitGB when switching to a non-Apple GPU
            if (newGpu?.vendor !== 'apple' && hardware.appleWiredLimitGB !== undefined) {
              patch.appleWiredLimitGB = undefined;
            }
            onChange(patch);
          }}
        >
          {VENDORS.map((v) => (
            <optgroup key={v.vendor} label={v.label}>
              {GPU_PRESETS.filter((g) => g.vendor === v.vendor).map((g) => (
                <option key={g.name} value={g.name}>
                  {g.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
      <div className="grid2">
        <NumberField label="VRAM per GPU" suffix="GB" value={hardware.vramGB} min={0.1} max={4096} step={1} onChange={(v) => onChange({ vramGB: v })} />
        <NumberField
          label="Bandwidth per GPU"
          suffix="GB/s"
          value={hardware.bandwidthGBs}
          min={1}
          max={100000}
          step={1}
          onChange={(v) => onChange({ bandwidthGBs: v })}
        />
      </div>
      <Stepper
        label="GPU count"
        value={hardware.gpuCount}
        min={1}
        max={MAX_GPUS}
        onChange={(v) => onChange({ gpuCount: v })}
        help="Multi-GPU assumes tensor parallel: VRAM and bandwidth add."
      />
      <div className="grid2">
        <NumberField
          label="Reserve"
          suffix="%"
          value={hardware.reservePct}
          min={0}
          max={50}
          step={1}
          onChange={(v) => onChange({ reservePct: v })}
          help="VRAM kept free (vLLM's gpu_memory_utilization 0.95 = 5%)."
        />
        <NumberField
          label="Overhead per GPU"
          suffix="GB"
          value={hardware.overheadGB}
          min={0}
          max={8}
          step={0.1}
          onChange={(v) => onChange({ overheadGB: v })}
          help="CUDA context, activations, runtime buffers."
        />
      </div>
      {findGpuPreset(hardware.gpuName)?.vendor === 'apple' && (
        <details className="panel-details">
          <summary>Apple wired-memory limit</summary>
          <p>
            macOS caps GPU-wired memory: 0.67× RAM for ≤36 GB, 0.75× above. Raise it with{' '}
            <code>sudo sysctl iogpu.wired_limit_mb=...</code> — headroom will use your override here if set.
          </p>
          <div className="field">
            <label htmlFor={`${gpuId}-awl`}>GPU-wired limit (leave blank for default)</label>
            <input
              id={`${gpuId}-awl`}
              type="number"
              min="0.1"
              max="4096"
              step="1"
              value={hardware.appleWiredLimitGB ?? ''}
              onChange={(e) => {
                const v = e.target.value.trim();
                onChange({ appleWiredLimitGB: v === '' ? undefined : Math.max(0.1, Math.min(4096, Number(v))) });
              }}
              placeholder="e.g. 120"
            />
            <small>Custom GPU-wired memory limit in GB. Leave blank to use macOS default.</small>
          </div>
        </details>
      )}
      <OffloadFields hardware={hardware} onChange={onChange} />
    </section>
  );
}
