// Planner tab state. Deliberately separate from the Calculator's reducer (src/ui/state.ts) so
// switching tabs never loses either side's work: Calculator <-> Planner <-> Calculator round
// trips both untouched.
//
// Contract for #24 (TaskPicker) and #25 (HardwareSizing): each step gets its own optional
// sub-state slice here (`taskPicker` / `hardwareSizing`) plus its own action(s) below, instead of
// widening a shared shape. `handoffModelId` is the one field both steps already share — step 1
// ("Which model for this task?") sets it via `setHandoffModelId`, step 2 ("What hardware to
// serve N users?") reads it to preselect the model it sizes hardware for.

import { DEFAULT_TASK_PICKER_CONSTRAINTS } from '../../lib';
import type { TaskPickerConstraints } from '../../lib';

export interface PlannerState {
  /** Set by TaskPicker (#24) once it recommends a model; read by HardwareSizing (#25). */
  handoffModelId?: string;
  /** The TaskPicker's own inputs (#24); absent until the user changes one, so a fresh Planner visit stays out of the URL. */
  taskPicker?: TaskPickerConstraints;
  /** #25 adds its own sizing state here (target users, chosen GPU, etc.) as an optional sub-object. */
  hardwareSizing?: Record<string, never>;
}

export const initialPlannerState: PlannerState = {};

export type PlannerAction =
  | { type: 'setHandoffModelId'; modelId: string | undefined }
  | { type: 'taskPicker/setConstraints'; patch: Partial<TaskPickerConstraints> }
  | { type: 'clear' };

export function plannerReducer(state: PlannerState, action: PlannerAction): PlannerState {
  switch (action.type) {
    case 'setHandoffModelId': {
      if (action.modelId === undefined) {
        const { handoffModelId: _drop, ...rest } = state;
        return rest;
      }
      return { ...state, handoffModelId: action.modelId };
    }
    case 'taskPicker/setConstraints': {
      const base = state.taskPicker ?? DEFAULT_TASK_PICKER_CONSTRAINTS;
      return { ...state, taskPicker: { ...base, ...action.patch } };
    }
    case 'clear':
      return {};
  }
}
