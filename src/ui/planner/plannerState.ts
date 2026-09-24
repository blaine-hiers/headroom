// Planner tab state. Deliberately separate from the Calculator's reducer (src/ui/state.ts) so
// switching tabs never loses either side's work: Calculator <-> Planner <-> Calculator round
// trips both untouched.
//
// Contract for #24 (TaskPicker) and #25 (HardwareSizing): each step gets its own optional
// sub-state slice here (`taskPicker` / `hardwareSizing`) plus its own action(s) below, instead of
// widening a shared shape. `handoffModelId` is the one field both steps already share — step 1
// ("Which model for this task?") sets it via `setHandoffModelId`, step 2 ("What hardware to
// serve N users?") reads it to preselect the model it sizes hardware for.

export interface PlannerState {
  /** Set by TaskPicker (#24) once it recommends a model; read by HardwareSizing (#25). */
  handoffModelId?: string;
  /** #24 adds its own picker state here (selections, filters, etc.) as an optional sub-object. */
  taskPicker?: Record<string, never>;
  /** #25 adds its own sizing state here (target users, chosen GPU, etc.) as an optional sub-object. */
  hardwareSizing?: Record<string, never>;
}

export const initialPlannerState: PlannerState = {};

export type PlannerAction =
  | { type: 'setHandoffModelId'; modelId: string | undefined }
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
    case 'clear':
      return {};
  }
}
