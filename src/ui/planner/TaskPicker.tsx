import type { Dispatch } from 'react';
import type { OpenInCalculatorPatch } from '../state';
import type { PlannerAction, PlannerState } from './plannerState';

interface Props {
  planner: PlannerState;
  dispatch: Dispatch<PlannerAction>;
  /** Loads a partial config into the Calculator's primary column and switches to it. */
  openInCalculator: (patch: OpenInCalculatorPatch) => void;
}

/**
 * Step 1 of the Planner: "Which model for this task?" (issue #24 fills this in). It will read
 * and write `planner.handoffModelId` via `dispatch({ type: 'setHandoffModelId', modelId })` so
 * step 2 (HardwareSizing, #25) can pick up the recommended model, and call `openInCalculator`
 * with `{ model, quant }` for its "use this" action.
 */
export function TaskPicker(_props: Props) {
  return (
    <section className="panel" aria-label="Which model for this task?">
      <h2>Which model for this task?</h2>
      <p className="help">Coming soon: pick a task and get a recommended model. (#24)</p>
    </section>
  );
}
