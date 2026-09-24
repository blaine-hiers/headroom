import { useCallback, useEffect, useMemo, useReducer } from 'react';
import { calculate, encodeState } from '../lib';
import type { CalcState, WeightQuantKey } from '../lib';
import { HardwarePanel } from './HardwarePanel';
import { Header } from './Header';
import { ModelPanel } from './ModelPanel';
import { QuantPanel } from './QuantPanel';
import { Results } from './Results';
import { initialState, maxContextFor, reducer } from './state';
import { WorkloadPanel } from './WorkloadPanel';

const URL_DEBOUNCE_MS = 250;

function urlFor(state: CalcState): string {
  return `${window.location.pathname}?${encodeState(state)}`;
}

function writeUrl(state: CalcState): void {
  try {
    window.history.replaceState(null, '', urlFor(state));
  } catch {
    // history can throw in sandboxed frames; the link just will not update
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, () => initialState(window.location.search));
  const result = useMemo(() => calculate(state), [state]);

  useEffect(() => {
    const t = window.setTimeout(() => writeUrl(state), URL_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [state]);

  const getLink = useCallback(() => {
    writeUrl(state);
    return `${window.location.origin}${urlFor(state)}`;
  }, [state]);

  const applyFit = useCallback((weight: WeightQuantKey, contextTokens: number) => {
    dispatch({ type: 'quant', patch: { weight } });
    dispatch({ type: 'workload', patch: { contextTokens } });
  }, []);

  return (
    <div className="app">
      <Header getLink={getLink} />
      <main className="layout">
        <div className="inputs">
          <ModelPanel
            model={state.model}
            onLoad={(spec) => dispatch({ type: 'loadModel', spec })}
            onEdit={(patch) => dispatch({ type: 'editModel', patch })}
          />
          <QuantPanel quant={state.quant} onChange={(patch) => dispatch({ type: 'quant', patch })} />
          <HardwarePanel
            hardware={state.hardware}
            runtime={state.runtime ?? 'generic'}
            onChange={(patch) => dispatch({ type: 'hardware', patch })}
            onRuntimeChange={(runtime) => dispatch({ type: 'runtime', runtime })}
          />
          <WorkloadPanel
            workload={state.workload}
            maxContext={maxContextFor(state.model)}
            onChange={(patch) => dispatch({ type: 'workload', patch })}
          />
        </div>
        <Results state={state} result={result} onApplyFit={applyFit} />
      </main>
      <footer className="footer muted">
        Model data from the Hugging Face Hub (config.json + safetensors parameter count, repo file sizes, GGUF headers) or built-in presets.
      </footer>
    </div>
  );
}
