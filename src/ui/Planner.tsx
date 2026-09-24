import type { Dispatch } from 'react';
import type { ModelSpec } from '../lib';
import type { OpenInCalculatorPatch } from './state';
import { HardwareSizing } from './planner/HardwareSizing';
import type { PlannerAction, PlannerState } from './planner/plannerState';
import { TaskPicker } from './planner/TaskPicker';

interface Props {
  planner: PlannerState;
  dispatch: Dispatch<PlannerAction>;
  /** Loads a partial config into the Calculator's primary column and switches to it. */
  openInCalculator: (patch: OpenInCalculatorPatch) => void;
  /** The Calculator's current primary-column model, for HardwareSizing's "use the Calculator's model" option (#25). */
  calculatorModel?: ModelSpec;
}

/**
 * The Planner tab shell (issue #21). It owns no sizing logic itself: step 1 (TaskPicker, #24)
 * and step 2 (HardwareSizing, #25) are separate components/files so those issues can land
 * without touching this file or each other. `.planner-toolbar` is the obvious slot #26's Clear
 * button goes in.
 */
export function Planner({ planner, dispatch, openInCalculator, calculatorModel }: Props) {
  return (
    <div className="planner">
      <div className="planner-toolbar" />
      <TaskPicker planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} />
      <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} calculatorModel={calculatorModel} />
    </div>
  );
}
