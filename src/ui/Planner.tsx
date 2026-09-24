import type { Dispatch } from 'react';
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
}

/**
 * The Planner tab shell (issue #21). It owns no sizing logic itself: step 1 (TaskPicker, #24)
 * and step 2 (HardwareSizing, #25) are separate components/files so those issues can land
 * without touching this file or each other. `.planner-toolbar` is the obvious slot #26's Clear
 * button goes in.
 */
export function Planner({ planner, dispatch, openInCalculator, onClear, showUndo, onUndo }: Props) {
  return (
    <div className="planner">
      {showUndo && onUndo && (
        <div className="undo-notice" role="status">
          <span className="muted">Cleared · </span>
          <button className="link-btn" onClick={onUndo}>
            Undo
          </button>
        </div>
      )}
      <div className="planner-toolbar">
        <button className="btn" onClick={onClear} aria-label="Clear planner">
          Clear
        </button>
      </div>
      <TaskPicker planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} />
      <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} />
    </div>
  );
}
