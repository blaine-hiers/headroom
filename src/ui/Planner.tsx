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
  onClear?: () => void;
  showUndo?: boolean;
  onUndo?: () => void;
  /** The Calculator's current primary-column model, for HardwareSizing's "use the Calculator's model" option (#25). */
  calculatorModel?: ModelSpec;
}

/**
 * The Planner tab shell (issue #21). It owns no sizing logic itself: step 1 (TaskPicker, #24)
 * and step 2 (HardwareSizing, #25) are separate components/files so those issues can land
 * without touching this file or each other. The Clear button (#26) sits in `.toolbar` above them.
 */
export function Planner({ planner, dispatch, openInCalculator, onClear, showUndo, onUndo, calculatorModel }: Props) {
  return (
    <div className="planner">
      <div className="toolbar">
        {showUndo && onUndo && (
          <div className="undo-notice" role="status">
            <span className="muted">Cleared · </span>
            <button className="link-btn" onClick={onUndo}>
              Undo
            </button>
          </div>
        )}
        <button className="btn toolbar-end" onClick={onClear} aria-label="Clear planner">
          Clear
        </button>
      </div>
      <TaskPicker planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} />
      <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} calculatorModel={calculatorModel} />
    </div>
  );
}
