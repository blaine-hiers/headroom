import { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import {
  calculate,
  cloneState,
  decodeCompareColumns,
  decodeTab,
  decodeTaskPickerState,
  DISABLED_SPECULATIVE,
  encodeCompareState,
  encodeTaskPickerState,
  MAX_COMPARE_COLUMNS,
  setTabParam,
} from '../lib';
import type { CalcState, TabKey, WeightQuantKey } from '../lib';
import { CompareTable } from './CompareTable';
import { HardwareFinder } from './HardwareFinder';
import { HardwarePanel } from './HardwarePanel';
import { Header } from './Header';
import { ModelPanel } from './ModelPanel';
import { plannerReducer } from './planner/plannerState';
import type { PlannerState } from './planner/plannerState';
import { Planner } from './Planner';
import { QuantPanel } from './QuantPanel';
import { Results } from './Results';
import { SpeculativeFields } from './SpeculativeFields';
import { clampState, initialState, maxContextFor, reducer } from './state';
import type { Action, OpenInCalculatorPatch } from './state';
import { TabBar } from './TabBar';
import { WorkloadPanel } from './WorkloadPanel';

const URL_DEBOUNCE_MS = 250;

function urlFor(columns: readonly CalcState[], tab: TabKey, planner: PlannerState): string {
  const params = new URLSearchParams(encodeCompareState(columns));
  setTabParam(params, tab);
  encodeTaskPickerState(params, planner.taskPicker);
  return `${window.location.pathname}?${params.toString()}`;
}

function writeUrl(columns: readonly CalcState[], tab: TabKey, planner: PlannerState): void {
  try {
    window.history.replaceState(null, '', urlFor(columns, tab, planner));
  } catch {
    // history can throw in sandboxed frames; the link just will not update
  }
}

function columnSummary(s: CalcState): string {
  return `${s.model.name} · ${s.hardware.gpuCount}× ${s.hardware.gpuName}`;
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, () => initialState(window.location.search));
  // decodeCompareColumns only validates shape/types; a crafted c2/c3 (e.g. a huge gpuCount or
  // negative vramGB) still needs the same range clamps the primary column gets via initialState.
  const [extraColumns, setExtraColumns] = useState<CalcState[]>(() => decodeCompareColumns(window.location.search).map(clampState));
  const [compareOn, setCompareOn] = useState(() => extraColumns.length > 0);
  const [selected, setSelected] = useState(0); // 0 = primary state, n = extraColumns[n - 1]
  const [tab, setTab] = useState<TabKey>(() => decodeTab(window.location.search));
  // Its own reducer, deliberately separate from the Calculator's: switching tabs never touches
  // or resets this, so Calculator -> Planner -> Calculator round-trips both untouched.
  const [planner, plannerDispatch] = useReducer(plannerReducer, undefined, () => {
    const taskPicker = decodeTaskPickerState(window.location.search);
    return taskPicker ? { taskPicker } : {};
  });

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
    const t = window.setTimeout(() => writeUrl(columns, tab, planner), URL_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [columns, tab, planner]);

  const getLink = useCallback(() => {
    writeUrl(columns, tab, planner);
    return `${window.location.origin}${urlFor(columns, tab, planner)}`;
  }, [columns, tab, planner]);

  /** Lets another tab (the Planner's "use this" actions) load a config into the Calculator's
   *  primary column and switch to it. Always targets the primary column, never whatever compare
   *  column happened to be selected, since that is what "open in Calculator" means. */
  const openInCalculator = useCallback((patch: OpenInCalculatorPatch) => {
    dispatch({ type: 'loadPartial', patch });
    setSelected(0);
    setTab('calculator');
  }, []);

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

  const applyFit = useCallback(
    (weight: WeightQuantKey, contextTokens: number) => {
      activeDispatch({ type: 'quant', patch: { weight } });
      activeDispatch({ type: 'workload', patch: { contextTokens } });
    },
    [activeDispatch],
  );

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
      <TabBar active={tab} onChange={setTab} />
      <div id="tabpanel-calculator" role="tabpanel" aria-labelledby="tab-calculator" hidden={tab !== 'calculator'}>
        <main className="layout">
          <div className="inputs">
            {/* Keyed by column: the GGUF picker, fetch status and search text are per-column local
                state, so switching columns must remount rather than carry B's picker over to A. */}
            <ModelPanel
              key={selected}
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
            <HardwareFinder state={activeState} onApply={(patch) => activeDispatch({ type: 'hardware', patch })} />
            <WorkloadPanel
              workload={activeState.workload}
              maxContext={maxContextFor(activeState.model)}
              onChange={(patch) => activeDispatch({ type: 'workload', patch })}
            />
            <SpeculativeFields
              speculative={activeState.speculative ?? DISABLED_SPECULATIVE}
              onChange={(patch) => activeDispatch({ type: 'speculative', patch })}
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
            <Results state={activeState} result={activeResult} onApplyFit={applyFit} getLink={getLink} />
          </div>
        </main>
      </div>
      <div id="tabpanel-planner" role="tabpanel" aria-labelledby="tab-planner" hidden={tab !== 'planner'}>
        <Planner planner={planner} dispatch={plannerDispatch} openInCalculator={openInCalculator} />
      </div>
      <footer className="footer muted">
        Model data from the Hugging Face Hub (config.json + safetensors parameter count, repo file sizes, GGUF headers) or built-in presets.
      </footer>
    </div>
  );
}
