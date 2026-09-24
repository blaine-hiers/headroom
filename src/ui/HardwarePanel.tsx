import { useId } from 'react';
import { GPU_PRESETS, RUNTIME_KEYS, RUNTIME_PROFILES, findGpuPreset } from '../lib';
import type { GpuVendor, HardwareSpec, RuntimeKey } from '../lib';
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
  runtime: RuntimeKey;
  onChange: (patch: Partial<HardwareSpec>) => void;
  onRuntimeChange: (runtime: RuntimeKey) => void;
}

/** Runtimes that replace the Reserve % field with their own fixed memory fraction. */
const RUNTIME_OVERRIDES_RESERVE: ReadonlySet<RuntimeKey> = new Set(['vllm', 'sglang']);

export function HardwarePanel({ hardware, runtime, onChange, onRuntimeChange }: Props) {
  const gpuId = useId();
  const runtimeId = useId();
  return (
    <section className="panel" aria-labelledby="hw-h">
      <h2 id="hw-h">Hardware</h2>
      <div className="field">
        <label htmlFor={runtimeId}>Runtime</label>
        <select id={runtimeId} value={runtime} onChange={(e) => onRuntimeChange(e.target.value as RuntimeKey)}>
          {RUNTIME_KEYS.map((key) => (
            <option key={key} value={key}>
              {RUNTIME_PROFILES[key].label}
            </option>
          ))}
        </select>
        <p className="help">{RUNTIME_PROFILES[runtime].blurb}</p>
      </div>
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
      <NumberField
        label="Compute per GPU (BF16)"
        suffix="TFLOPS"
        value={hardware.tflopsBf16}
        min={0.1}
        max={100000}
        step={0.1}
        onChange={(v) => onChange({ tflopsBf16: v })}
        help="Dense tensor TFLOPS, no sparsity. Used for the prefill/time-to-first-token estimate."
      />
      <Stepper
        label="GPU count"
        value={hardware.gpuCount}
        min={1}
        max={MAX_GPUS}
        onChange={(v) => onChange({ gpuCount: v })}
        help="Multi-GPU assumes tensor parallel: VRAM and bandwidth add."
      />
      <div className="field">
        <label htmlFor={`${gpuId}-cost`}>Cloud cost</label>
        <div className="input-wrap">
          <input
            id={`${gpuId}-cost`}
            type="number"
            min="0.01"
            max="1000"
            step="0.01"
            value={hardware.usdPerHour ?? ''}
            onChange={(e) => {
              const v = e.target.value.trim();
              onChange({ usdPerHour: v === '' ? undefined : Math.max(0.01, Math.min(1000, Number(v))) });
            }}
            placeholder="e.g. 3.25"
          />
          <span className="suffix">$/GPU-hr</span>
        </div>
        <p className="help">
          Typical on-demand list price, pre-filled from the preset (prices as of 2026-09, approximate). Leave blank to hide the cost card.
        </p>
      </div>
      <div className="grid2">
        <NumberField
          label="Reserve"
          suffix="%"
          value={hardware.reservePct}
          min={0}
          max={50}
          step={1}
          onChange={(v) => onChange({ reservePct: v })}
          help={
            RUNTIME_OVERRIDES_RESERVE.has(runtime)
              ? `Ignored: ${RUNTIME_PROFILES[runtime].label} uses its own fixed memory fraction instead (see the runtime selector above).`
              : "VRAM kept free (vLLM's gpu_memory_utilization 0.95 = 5%)."
          }
        />
        <NumberField
          label="Overhead per GPU"
          suffix="GB"
          value={hardware.overheadGB}
          min={0}
          max={8}
          step={0.1}
          onChange={(v) => onChange({ overheadGB: v })}
          help={
            runtime === 'vllm'
              ? 'CUDA context, activations, runtime buffers, plus a 1 GB CUDA-graph allowance vLLM adds on top.'
              : 'CUDA context, activations, runtime buffers.'
          }
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
