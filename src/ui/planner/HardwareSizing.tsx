import { useId, useMemo, useState } from 'react';
import type { Dispatch } from 'react';
import {
  DEFAULT_OFFLOAD,
  findModelPreset,
  formatNumber,
  formatSeconds,
  formatUsd,
  GPU_VENDOR_GROUPS,
  KV_QUANTS,
  MODEL_PRESETS,
  RUNTIME_KEYS,
  RUNTIME_PROFILES,
  scalingStrip,
  sizeHardware,
  WEIGHT_QUANTS,
} from '../../lib';
import type { GpuVendor, HardwareSizingOptions, HardwareSizingRow, HardwareSizingSort, KvQuantKey, ModelSpec, RuntimeKey, WeightQuantKey } from '../../lib';
import { Bytes } from '../Bytes';
import { NumberField } from '../NumberField';
import type { OpenInCalculatorPatch } from '../state';
import { MAX_USERS, maxContextFor, MIN_CONTEXT } from '../state';
import { DEFAULT_HARDWARE_SIZING } from './plannerState';
import type { HardwareSizingState, PlannerAction, PlannerState } from './plannerState';

interface Props {
  planner: PlannerState;
  dispatch: Dispatch<PlannerAction>;
  /** Loads a partial config into the Calculator's primary column and switches to it. */
  openInCalculator: (patch: OpenInCalculatorPatch) => void;
  /** The Calculator's current primary-column model, for the "use the Calculator's model" option. */
  calculatorModel?: ModelSpec;
}

type ModelSource = 'calculator' | 'catalog' | 'handoff' | 'default';

function resolveModel(hs: HardwareSizingState, planner: PlannerState, calculatorModel: ModelSpec | undefined): { model: ModelSpec; source: ModelSource } {
  if (hs.useCalculatorModel && calculatorModel) return { model: calculatorModel, source: 'calculator' };
  if (hs.modelId) {
    const m = findModelPreset(hs.modelId);
    if (m) return { model: m, source: 'catalog' };
  }
  if (planner.handoffModelId) {
    const m = findModelPreset(planner.handoffModelId);
    if (m) return { model: m, source: 'handoff' };
  }
  return { model: MODEL_PRESETS[0], source: 'default' };
}

function HwSizingRow({ row, dimmed, onUse }: { row: HardwareSizingRow; dimmed: boolean; onUse: (row: HardwareSizingRow) => void }) {
  const t = row.result.throughput;
  return (
    <tr className={dimmed ? 'hwsizing-dim' : undefined}>
      <td>
        {row.gpuCount}× {row.gpu.name}
        {row.unusual && (
          <span className="tag" title="More than 2 consumer GPUs is an unusual multi-GPU setup.">
            unusual
          </span>
        )}
      </td>
      <td className="num">{formatNumber(row.totalVramGB)} GB</td>
      <td>
        <Bytes value={row.result.runHeadroomBytes} />
      </td>
      <td className="num">{formatNumber(t.perUserTokS, t.perUserTokS < 10 ? 1 : 0)}</td>
      <td className="num">{formatNumber(t.aggregateTokS)}</td>
      <td className="num">{formatSeconds(row.result.prefill.ttftSeconds)}</td>
      <td className="num">{Number.isFinite(row.result.maxUsersAtContext) ? formatNumber(row.result.maxUsersAtContext) : '—'}</td>
      <td className="num">{row.cost ? formatUsd(row.cost.costPerHour) : '—'}</td>
      <td className="num">{row.cost?.atCurrentUsers !== undefined ? formatUsd(row.cost.atCurrentUsers) : '—'}</td>
      <td>
        {row.qualifies ? (
          <button type="button" className="btn btn-primary" onClick={() => onUse(row)}>
            Use
          </button>
        ) : (
          <span className="muted">{row.failReason}</span>
        )}
      </td>
    </tr>
  );
}

/**
 * Step 2 of the Planner: "What hardware to serve N users?" (issue #25). Reads
 * `planner.handoffModelId` (set by TaskPicker, #24) to preselect the model handed over from step
 * 1 when the user hasn't picked one here or asked for the Calculator's current model.
 */
export function HardwareSizing({ planner, dispatch, openInCalculator, calculatorModel }: Props) {
  const hs = planner.hardwareSizing ?? DEFAULT_HARDWARE_SIZING;
  const patch = (p: Partial<HardwareSizingState>) => dispatch({ type: 'hardwareSizing/patch', patch: p });

  const { model, source } = useMemo(() => resolveModel(hs, planner, calculatorModel), [hs, planner, calculatorModel]);
  const [modelNotFound, setModelNotFound] = useState(false);

  const modelListId = useId();
  const weightQuantId = useId();
  const kvQuantId = useId();
  const runtimeId = useId();
  const vendorId = useId();
  const sortId = useId();

  function commitModelText(text: string) {
    const match = findModelPreset(text);
    if (match) {
      setModelNotFound(false);
      patch({ modelId: match.id, useCalculatorModel: false });
    } else if (text.trim() !== model.name) {
      setModelNotFound(true);
    }
  }

  const options: HardwareSizingOptions = useMemo(
    () => ({
      quant: { weight: hs.weightQuant, kv: hs.kvQuant },
      runtime: hs.runtime,
      minPerUserTokS: hs.minPerUserTokS,
      maxTtftSeconds: hs.maxTtftSeconds,
      vendor: hs.vendor,
      offload: hs.offloadEnabled ? { enabled: true, systemRamGB: DEFAULT_OFFLOAD.systemRamGB, ramBandwidthGBs: DEFAULT_OFFLOAD.ramBandwidthGBs } : undefined,
      sort: hs.sort,
    }),
    [hs.weightQuant, hs.kvQuant, hs.runtime, hs.minPerUserTokS, hs.maxTtftSeconds, hs.vendor, hs.offloadEnabled, hs.sort],
  );

  const load = useMemo(() => ({ contextTokens: hs.contextTokens, concurrentUsers: hs.concurrentUsers }), [hs.contextTokens, hs.concurrentUsers]);
  const { qualifying, nearMisses } = useMemo(() => sizeHardware(model, load, options), [model, load, options]);
  const strip = useMemo(() => scalingStrip(model, hs.contextTokens, options), [model, hs.contextTokens, options]);

  const useRow = (row: HardwareSizingRow) => openInCalculator(row.state);

  return (
    <section className="panel" aria-label="What hardware to serve N users?">
      <h2>What hardware to serve N users?</h2>

      <div className="field">
        <label htmlFor={`${modelListId}-input`}>Model</label>
        <input
          id={`${modelListId}-input`}
          key={model.id}
          type="text"
          list={modelListId}
          defaultValue={model.name}
          onBlur={(e) => commitModelText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitModelText((e.target as HTMLInputElement).value);
          }}
        />
        <datalist id={modelListId}>
          {MODEL_PRESETS.map((p) => (
            <option key={p.id} value={p.name} />
          ))}
        </datalist>
      </div>
      {modelNotFound && <p className="help">Not in the catalog — pick a model from the list or spell it exactly.</p>}
      {source === 'handoff' && <p className="help">Using {model.name}, handed over from step 1.</p>}
      {source === 'calculator' && <p className="help">Using {model.name}, the Calculator's current model.</p>}
      <label className="check">
        <input
          type="checkbox"
          checked={hs.useCalculatorModel}
          disabled={!calculatorModel}
          onChange={(e) => patch({ useCalculatorModel: e.target.checked })}
        />
        Use the Calculator's current model{calculatorModel ? ` (${calculatorModel.name})` : ''}
      </label>

      <div className="grid2">
        <NumberField
          label="Users to serve"
          value={hs.concurrentUsers}
          min={1}
          max={MAX_USERS}
          step={1}
          integer
          onChange={(v) => patch({ concurrentUsers: v })}
        />
        <NumberField
          label="Context per user"
          suffix="tok"
          value={hs.contextTokens}
          min={MIN_CONTEXT}
          max={maxContextFor(model)}
          step={256}
          integer
          onChange={(v) => patch({ contextTokens: v })}
        />
      </div>

      <div className="grid2">
        <div className="field">
          <label htmlFor={weightQuantId}>Weight quant</label>
          <select id={weightQuantId} value={hs.weightQuant} onChange={(e) => patch({ weightQuant: e.target.value as WeightQuantKey })}>
            {Object.entries(WEIGHT_QUANTS).map(([k, info]) => (
              <option key={k} value={k}>
                {info.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={kvQuantId}>KV dtype</label>
          <select id={kvQuantId} value={hs.kvQuant} onChange={(e) => patch({ kvQuant: e.target.value as KvQuantKey })}>
            {Object.entries(KV_QUANTS).map(([k, info]) => (
              <option key={k} value={k}>
                {info.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid2">
        <div className="field">
          <label htmlFor={runtimeId}>Serving runtime</label>
          <select id={runtimeId} value={hs.runtime} onChange={(e) => patch({ runtime: e.target.value as RuntimeKey })}>
            {RUNTIME_KEYS.map((k) => (
              <option key={k} value={k}>
                {RUNTIME_PROFILES[k].label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={vendorId}>Vendor filter</label>
          <select
            id={vendorId}
            value={hs.vendor ?? 'all'}
            onChange={(e) => patch({ vendor: e.target.value === 'all' ? undefined : (e.target.value as GpuVendor) })}
          >
            <option value="all">All vendors</option>
            {GPU_VENDOR_GROUPS.map((v) => (
              <option key={v.vendor} value={v.vendor}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid2">
        <NumberField
          label="Min per-user decode"
          suffix="tok/s"
          value={hs.minPerUserTokS}
          min={0}
          max={1000}
          step={1}
          onChange={(v) => patch({ minPerUserTokS: v })}
        />
        <div className="field">
          <label className="check">
            <input
              type="checkbox"
              checked={hs.maxTtftSeconds !== undefined}
              onChange={(e) => patch({ maxTtftSeconds: e.target.checked ? 2 : undefined })}
            />
            Cap time-to-first-token
          </label>
          {hs.maxTtftSeconds !== undefined && (
            <NumberField
              label="Max TTFT"
              suffix="s"
              value={hs.maxTtftSeconds}
              min={0.05}
              max={60}
              step={0.1}
              decimals={2}
              onChange={(v) => patch({ maxTtftSeconds: v })}
            />
          )}
        </div>
      </div>

      <label className="check">
        <input type="checkbox" checked={hs.offloadEnabled} onChange={(e) => patch({ offloadEnabled: e.target.checked })} />
        Allow CPU/RAM offload for layers that don't fit
      </label>

      <div className="field">
        <label htmlFor={sortId}>Sort qualifying rows by</label>
        <select id={sortId} value={hs.sort} onChange={(e) => patch({ sort: e.target.value as HardwareSizingSort })}>
          <option value="smallest">Smallest first (total VRAM)</option>
          <option value="cheapest">Cheapest cloud $/hr</option>
        </select>
      </div>

      {qualifying.length === 0 && nearMisses.length === 0 ? (
        <p className="muted">Nothing in the bundled GPU table fits at 1, 2, 4 or 8 GPUs for this model, quant and workload.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">GPU × count</th>
                <th scope="col">Total VRAM</th>
                <th scope="col">Headroom</th>
                <th scope="col">Tok/s /user</th>
                <th scope="col">Tok/s total</th>
                <th scope="col">TTFT</th>
                <th scope="col">Max users</th>
                <th scope="col">$/hr</th>
                <th scope="col">$/1M tok</th>
                <th scope="col"></th>
              </tr>
            </thead>
            <tbody>
              {qualifying.map((row) => (
                <HwSizingRow key={`${row.gpu.name}-${row.gpuCount}`} row={row} dimmed={false} onUse={useRow} />
              ))}
              {nearMisses.map((row) => (
                <HwSizingRow key={`miss-${row.gpu.name}-${row.gpuCount}`} row={row} dimmed onUse={useRow} />
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="help">
        {hs.sort === 'cheapest'
          ? 'Ranked by $/hour where a price is listed, otherwise by total VRAM.'
          : 'Ranked by total VRAM, smallest first, regardless of price.'}{' '}
        Dimmed rows are near-misses shown with why they fall short.
      </p>

      <div className="table-wrap">
        <table>
          <caption className="help scaling-strip-caption">
            How does this scale? Smallest qualifying option for {model.name} at {formatNumber(hs.contextTokens)} tok context.
          </caption>
          <thead>
            <tr>
              {strip.map((entry) => (
                <th scope="col" key={entry.users}>
                  {formatNumber(entry.users)} users
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              {strip.map((entry) => (
                <td key={entry.users}>
                  {entry.row ? (
                    <>
                      {entry.row.gpuCount}× {entry.row.gpu.name}
                      {entry.row.cost && <div className="muted">{formatUsd(entry.row.cost.costPerHour)}/hr</div>}
                    </>
                  ) : (
                    <span className="muted">none fit</span>
                  )}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
