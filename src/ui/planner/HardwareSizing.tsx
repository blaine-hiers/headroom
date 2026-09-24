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
 * Step 2 of the Planner: "What hardware to serve N users?" (issue #25 fills this in). It will
 * read `planner.handoffModelId` (set by TaskPicker, #24) to preselect the model it sizes
 * hardware for, and call `openInCalculator` with `{ model, hardware, workload }` for its
 * "use this" action.
 */
export function HardwareSizing(_props: Props) {
  return (
    <section className="panel" aria-label="What hardware to serve N users?">
      <h2>What hardware to serve N users?</h2>
      <p className="help">Coming soon: size hardware for a target user count. (#25)</p>
    </section>
  );
}
