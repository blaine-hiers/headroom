import { useId, useMemo } from 'react';
import type { Dispatch } from 'react';
import {
  CUSTOM_GPU_NAME,
  DEFAULT_TASK_PICKER_CONSTRAINTS,
  formatNumber,
  formatSeconds,
  GPU_PRESETS,
  KV_QUANTS,
  MODEL_CATALOG,
  PERMISSIVE_LICENSES,
  PROVIDERS,
  rankModelsForTask,
  TASK_PICKER_RULE,
  TASK_TAGS,
  WEIGHT_QUANTS,
} from '../../lib';
import type { KvQuantKey, TaskPickerConstraints, WeightQuantKey } from '../../lib';
import type { OpenInCalculatorPatch } from '../state';
import { MAX_GPUS, MAX_USERS, MIN_CONTEXT } from '../state';
import { Bytes } from '../Bytes';
import type { PlannerAction, PlannerState } from './plannerState';

interface Props {
  planner: PlannerState;
  dispatch: Dispatch<PlannerAction>;
  /** Loads a partial config into the Calculator's primary column and switches to it. */
  openInCalculator: (patch: OpenInCalculatorPatch) => void;
}

/** Task labels are the tag itself, title-cased with the hyphen turned into a space. */
function taskLabel(tag: string): string {
  return tag
    .split('-')
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(' ');
}

const billions = (n: number) => `${formatNumber(n / 1e9, n < 1e9 ? 3 : 1)}B`;

/**
 * Step 1 of the Planner: "Which model for this task?" (#24). Picks a task and hardware/quant/
 * license/MoE constraints, ranks the bundled catalog with `rankModelsForTask`, and lets each row
 * either open in the Calculator or hand off to step 2 (HardwareSizing, #25) via `handoffModelId`.
 */
export function TaskPicker({ planner, dispatch, openInCalculator }: Props) {
  const constraints = planner.taskPicker ?? DEFAULT_TASK_PICKER_CONSTRAINTS;
  const taskId = useId();
  const gpuId = useId();
  const gpuCountId = useId();
  const contextId = useId();
  const usersId = useId();
  const weightQuantId = useId();
  const kvQuantId = useId();
  const licenseId = useId();

  const setConstraints = (patch: Partial<TaskPickerConstraints>) => dispatch({ type: 'taskPicker/setConstraints', patch });

  const { rows, hint } = useMemo(() => rankModelsForTask(MODEL_CATALOG, constraints), [constraints]);

  return (
    <section className="panel step" aria-label="Which model for this task?">
      <h2>
        <span className="step-num" aria-hidden="true">
          1
        </span>
        <span>Which model for this task?</span>
      </h2>
      <div className="controls">
        <div className="field">
          <label htmlFor={taskId}>Task</label>
          <select id={taskId} value={constraints.task} onChange={(e) => setConstraints({ task: e.target.value as TaskPickerConstraints['task'] })}>
            {TASK_TAGS.map((tag) => (
              <option key={tag} value={tag}>
                {taskLabel(tag)}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={gpuId}>Hardware</label>
          <select id={gpuId} value={constraints.gpuName} onChange={(e) => setConstraints({ gpuName: e.target.value })}>
            <option value="any">Any GPU (search the bundled table)</option>
            {GPU_PRESETS.filter((g) => g.name !== CUSTOM_GPU_NAME).map((g) => (
              <option key={g.name} value={g.name}>
                {g.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={gpuCountId}>How many GPUs</label>
          <input
            id={gpuCountId}
            type="number"
            min={1}
            max={MAX_GPUS}
            step={1}
            value={constraints.gpuCount}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) setConstraints({ gpuCount: Math.round(Math.min(MAX_GPUS, Math.max(1, n))) });
            }}
          />
        </div>
        <div className="field">
          <label htmlFor={usersId}>How many concurrent users</label>
          <input
            id={usersId}
            type="number"
            min={1}
            max={MAX_USERS}
            step={1}
            value={constraints.concurrentUsers}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) setConstraints({ concurrentUsers: Math.round(Math.min(MAX_USERS, Math.max(1, n))) });
            }}
          />
        </div>
        <div className="field">
          <label htmlFor={contextId}>Context needed</label>
          <div className="input-wrap">
            <input
              id={contextId}
              type="number"
              min={MIN_CONTEXT}
              step={1024}
              value={constraints.contextTokens}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (Number.isFinite(n)) setConstraints({ contextTokens: Math.max(MIN_CONTEXT, Math.round(n)) });
              }}
            />
            <span className="suffix">tokens</span>
          </div>
        </div>
        <div className="field">
          <label htmlFor={weightQuantId}>Weight quant</label>
          <select id={weightQuantId} value={constraints.weightQuant} onChange={(e) => setConstraints({ weightQuant: e.target.value as WeightQuantKey })}>
            {Object.entries(WEIGHT_QUANTS).map(([key, info]) => (
              <option key={key} value={key}>
                {info.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={kvQuantId}>KV dtype</label>
          <select id={kvQuantId} value={constraints.kvQuant} onChange={(e) => setConstraints({ kvQuant: e.target.value as KvQuantKey })}>
            {Object.entries(KV_QUANTS).map(([key, info]) => (
              <option key={key} value={key}>
                {info.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={licenseId}>License</label>
          <select
            id={licenseId}
            value={constraints.licenseFilter}
            onChange={(e) => setConstraints({ licenseFilter: e.target.value as TaskPickerConstraints['licenseFilter'] })}
          >
            <option value="any">Any license</option>
            <option value="permissive">Permissive only ({PERMISSIVE_LICENSES.map((l) => l.toUpperCase()).join(', ')})</option>
          </select>
        </div>
        <label className="check controls-check">
          <input type="checkbox" checked={constraints.allowMoe} onChange={(e) => setConstraints({ allowMoe: e.target.checked })} /> Allow MoE models
        </label>
      </div>
      <p className="help">{TASK_PICKER_RULE}</p>
      {hint ? (
        <p className="muted">{hint}</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Provider</th>
                <th scope="col">Params</th>
                <th scope="col">Weights</th>
                <th scope="col">Fit</th>
                <th scope="col">Headroom</th>
                <th scope="col">Tok/s</th>
                <th scope="col">TTFT</th>
                <th scope="col">License</th>
                <th scope="col"></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const provider = PROVIDERS.find((p) => p.id === row.entry.meta.provider)?.name ?? row.entry.meta.provider;
                const params = row.entry.spec.moe ? row.entry.spec.activeParams : row.entry.spec.params;
                const fitWord = row.result.fits ? 'Fits' : row.result.runs ? 'Offload' : 'No fit';
                return (
                  <tr key={row.entry.spec.id}>
                    <td>
                      {row.entry.spec.name}
                      <p className="help row-reason">{row.reason}</p>
                    </td>
                    <td>{provider}</td>
                    <td className="num">
                      {billions(params)}
                      {row.entry.spec.moe && ' active'}
                    </td>
                    <td className="num">
                      <Bytes value={row.result.weightBytes} />
                    </td>
                    <td>
                      <span className={`fit-badge ${row.result.fits ? 'fit-badge-ok' : row.result.runs ? 'fit-badge-warn' : 'fit-badge-bad'}`}>{fitWord}</span>
                    </td>
                    <td className="num">
                      <Bytes value={row.result.headroomBytes} />
                    </td>
                    <td className="num">{formatNumber(row.result.throughput.perUserTokS, row.result.throughput.perUserTokS < 10 ? 1 : 0)}</td>
                    <td className="num">{formatSeconds(row.result.prefill.ttftSeconds)}</td>
                    <td>{row.entry.meta.license}</td>
                    <td>
                      <div className="stack-actions">
                        <button
                          type="button"
                          className="btn btn-ghost"
                          onClick={() =>
                            openInCalculator({
                              model: row.entry.spec,
                              quant: { weight: constraints.weightQuant, kv: constraints.kvQuant },
                              hardware: row.hardware,
                              workload: row.workload,
                              runtime: 'generic',
                            })
                          }
                        >
                          Use
                        </button>
                        <button type="button" className="btn btn-ghost" onClick={() => dispatch({ type: 'setHandoffModelId', modelId: row.entry.spec.id })}>
                          Size hardware
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
