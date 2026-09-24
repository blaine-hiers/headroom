import { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import { calculate, cloneState, encodeCompareState, decodeCompareColumns } from '../lib';
import type { CalcState } from '../lib';
import { CompareTable } from './CompareTable';
import { HardwarePanel } from './HardwarePanel';
import { Header } from './Header';
import { ModelPanel } from './ModelPanel';
import { QuantPanel } from './QuantPanel';
import { Results } from './Results';
import { initialState, maxContextFor, reducer } from './state';
import type { Action } from './state';
import { WorkloadPanel } from './WorkloadPanel';

const URL_DEBOUNCE_MS = 250;
const MAX_COMPARE_COLUMNS = 3;

function urlFor(columns: readonly CalcState[]): string {
  return `${window.location.pathname}?${encodeCompareState(columns)}`;
}

function writeUrl(columns: readonly CalcState[]): void {
  try {
    window.history.replaceState(null, '', urlFor(columns));
  } catch {
    // history can throw in sandboxed frames; the link just will not update
  }
}

function columnSummary(s: CalcState): string {
  return `${s.model.name} · ${s.hardware.gpuCount}× ${s.hardware.gpuName}`;
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, () => initialState(window.location.search));
  const [extraColumns, setExtraColumns] = useState<CalcState[]>(() => decodeCompareColumns(window.location.search));
  const [compareOn, setCompareOn] = useState(() => extraColumns.length > 0);
  const [selected, setSelected] = useState(0); // 0 = primary state, n = extraColumns[n - 1]

  const columns = useMemo(() => [state, ...extraColumns], [state, extraColumns]);
  const activeState = columns[selected] ?? state;
  const activeDispatch = useCallback(
    (action: Action) => {
      if (selected === 0) {
        dispatch(action);
      } else {
        setExtraColumns((cols) => cols.map((c, i) => (i === selected - 1 ? reducer(c, action) : c)));
      }
    },
    [selected],
  );

  const results = useMemo(() => columns.map((c) => calculate(c)), [columns]);
  const activeResult = results[selected] ?? results[0];

  useEffect(() => {
    const t = window.setTimeout(() => writeUrl(columns), URL_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [columns]);

  const getLink = useCallback(() => {
    writeUrl(columns);
    return `${window.location.origin}${urlFor(columns)}`;
  }, [columns]);

  const toggleCompare = useCallback(() => {
    setCompareOn((on) => {
      if (on) {
        setExtraColumns([]);
        setSelected(0);
        return false;
      }
      setExtraColumns((cols) => (cols.length > 0 ? cols : [cloneState(state)]));
      return true;
    });
  }, [state]);

  const duplicateColumn = useCallback(() => {
    setExtraColumns((cols) => {
      if (cols.length + 1 >= MAX_COMPARE_COLUMNS) return cols;
      const copy = cloneState(activeState);
      const next = [...cols, copy];
      setSelected(next.length); // select the new column (index into `columns`, 1-based among extras)
      return next;
    });
  }, [activeState]);

  const removeColumn = useCallback((index: number) => {
    if (index === 0) return; // the primary column can't be removed
    setExtraColumns((cols) => {
      const next = cols.filter((_, i) => i !== index - 1);
      if (next.length === 0) setCompareOn(false);
      return next;
    });
    setSelected((sel) => (sel === index ? 0 : sel > index ? sel - 1 : sel));
  }, []);

  return (
    <div className="app">
      <Header getLink={getLink} compareOn={compareOn} onToggleCompare={toggleCompare} />
      <main className="layout">
        <div className="inputs">
          <ModelPanel
            model={activeState.model}
            onLoad={(spec) => activeDispatch({ type: 'loadModel', spec })}
            onEdit={(patch) => activeDispatch({ type: 'editModel', patch })}
          />
          <QuantPanel quant={activeState.quant} onChange={(patch) => activeDispatch({ type: 'quant', patch })} />
          <HardwarePanel
            hardware={activeState.hardware}
            runtime={activeState.runtime ?? 'generic'}
            onChange={(patch) => activeDispatch({ type: 'hardware', patch })}
            onRuntimeChange={(runtime) => activeDispatch({ type: 'runtime', runtime })}
          />
          <WorkloadPanel
            workload={activeState.workload}
            maxContext={maxContextFor(activeState.model)}
            onChange={(patch) => activeDispatch({ type: 'workload', patch })}
          />
        </div>
        <div className="results-stack">
          {compareOn && (
            <CompareTable
              columns={columns.map((c, i) => ({ result: results[i], summary: columnSummary(c) }))}
              selected={selected}
              onSelect={setSelected}
              onDuplicate={duplicateColumn}
              onRemove={removeColumn}
            />
          )}
          <Results state={activeState} result={activeResult} />
        </div>
      </main>
      <footer className="footer muted">
        Model data from the Hugging Face Hub (config.json + safetensors parameter count, repo file sizes, GGUF headers) or built-in presets.
      </footer>
    </div>
  );
}
