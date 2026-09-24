import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useReducer } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { calculate, findModelPreset } from '../../lib';
import type { OpenInCalculatorPatch } from '../state';
import { HardwareSizing } from './HardwareSizing';
import { DEFAULT_HARDWARE_SIZING, initialPlannerState, plannerReducer } from './plannerState';
import type { PlannerState } from './plannerState';

function Harness({ openInCalculator }: { openInCalculator: (patch: OpenInCalculatorPatch) => void }) {
  const [planner, dispatch] = useReducer(plannerReducer, initialPlannerState);
  return <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} />;
}

describe('HardwareSizing', () => {
  it('sizes hardware for 32 users / 8K context / 20 tok/s (the defaults) and shows a results table', async () => {
    render(<Harness openInCalculator={vi.fn()} />);
    expect(screen.getByLabelText('Users to serve')).toHaveValue(32);
    expect(screen.getByLabelText('Context per user')).toHaveValue(8192);
    expect(screen.getByLabelText('Min per-user decode')).toHaveValue(20);
    // At least one qualifying row with a "Use" action, or the explicit "nothing fits" message —
    // either way the table renders without throwing.
    const useButtons = screen.queryAllByRole('button', { name: 'Use' });
    const nothingFits = screen.queryByText(/Nothing in the bundled GPU table fits/);
    expect(useButtons.length > 0 || nothingFits !== null).toBe(true);
  });

  it('"Use" loads the Calculator with the exact config the row was computed from, and it reproduces the same result', async () => {
    const user = userEvent.setup();
    const openInCalculator = vi.fn();
    render(<Harness openInCalculator={openInCalculator} />);

    const useButtons = await screen.findAllByRole('button', { name: 'Use' });
    expect(useButtons.length).toBeGreaterThan(0);
    await user.click(useButtons[0]);

    expect(openInCalculator).toHaveBeenCalledTimes(1);
    const patch = openInCalculator.mock.calls[0][0] as OpenInCalculatorPatch;
    expect(patch.model).toBeDefined();
    expect(patch.hardware).toBeDefined();
    expect(patch.workload).toEqual({ contextTokens: 8192, concurrentUsers: 32 });

    // The badge the Calculator would show agrees with what the row promised: it runs, and
    // per-user tok/s meets the 20 tok/s default target.
    const result = calculate(patch as Parameters<typeof calculate>[0]);
    expect(result.runs).toBe(true);
    expect(result.throughput.perUserTokS).toBeGreaterThanOrEqual(20);
  });

  it('defaults the sort control to "Smallest first" and lets it switch to "Cheapest cloud $/hr"', async () => {
    const user = userEvent.setup();
    render(<Harness openInCalculator={vi.fn()} />);
    const sortSelect = screen.getByLabelText('Sort qualifying rows by') as HTMLSelectElement;
    expect(sortSelect).toHaveValue('smallest');
    expect(screen.getByText(/Ranked by total VRAM, smallest first, regardless of price/)).toBeInTheDocument();

    await user.selectOptions(sortSelect, 'cheapest');
    expect(sortSelect).toHaveValue('cheapest');
    expect(screen.getByText(/Ranked by \$\/hour where a price is listed/)).toBeInTheDocument();
  });

  it('shows a note when the model was handed over from step 1', () => {
    function HandoffHarness({ openInCalculator }: { openInCalculator: (patch: OpenInCalculatorPatch) => void }) {
      const [planner, dispatch] = useReducer(plannerReducer, { handoffModelId: 'meta-llama/Llama-3.3-70B-Instruct' });
      return <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} />;
    }
    render(<HandoffHarness openInCalculator={vi.fn()} />);
    expect(screen.getByText(/handed over from step 1/)).toBeInTheDocument();
  });

  // Regression for #27 review item 3: "Size hardware" used to silently do nothing once step 2
  // already had an explicit model pick or "use the Calculator's model" on, because resolveModel
  // checked those before the handoff. A handoff (setHandoffModelId) must now win over both.
  it('"Size hardware" wins over an existing explicit pick and "use Calculator model" (#27 review item 3)', async () => {
    const user = userEvent.setup();
    const seventyB = findModelPreset('meta-llama/Llama-3.3-70B-Instruct')!;

    function OverrideHarness({ openInCalculator }: { openInCalculator: (patch: OpenInCalculatorPatch) => void }) {
      const initial: PlannerState = {
        hardwareSizing: { ...DEFAULT_HARDWARE_SIZING, modelId: seventyB.id, useCalculatorModel: true },
      };
      const [planner, dispatch] = useReducer(plannerReducer, initial);
      return (
        <>
          <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={openInCalculator} calculatorModel={seventyB} />
          <button type="button" onClick={() => dispatch({ type: 'setHandoffModelId', modelId: 'meta-llama/Llama-3.1-8B-Instruct' })}>
            simulate step 1 handoff
          </button>
        </>
      );
    }
    render(<OverrideHarness openInCalculator={vi.fn()} />);

    expect(screen.getByDisplayValue('Llama 3.3 70B')).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Use the Calculator's current model/ })).toBeChecked();

    await user.click(screen.getByRole('button', { name: 'simulate step 1 handoff' }));

    expect(screen.getByDisplayValue('Llama 3.1 8B')).toBeInTheDocument();
    expect(screen.getByText(/handed over from step 1/)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Use the Calculator's current model/ })).not.toBeChecked();
  });

  // Regression for #27 review item 1: switching to a shorter-context model left the typed
  // context above that model's max, so every row calculated at the raw, unclamped value instead
  // of what "Use" would actually load — the UI must surface that the two now disagree.
  it('shows a note and sizes at the model max when the typed context exceeds it (#27 review item 1)', async () => {
    const user = userEvent.setup();
    const openInCalculator = vi.fn();
    function Harness2({ onOpenInCalculator }: { onOpenInCalculator: (patch: OpenInCalculatorPatch) => void }) {
      const initial: PlannerState = { hardwareSizing: { ...DEFAULT_HARDWARE_SIZING, contextTokens: 131072 } };
      const [planner, dispatch] = useReducer(plannerReducer, initial);
      return <HardwareSizing planner={planner} dispatch={dispatch} openInCalculator={onOpenInCalculator} />;
    }
    render(<Harness2 onOpenInCalculator={openInCalculator} />);

    const modelInput = screen.getByLabelText('Model');
    await user.clear(modelInput);
    await user.type(modelInput, 'Phi-4');
    await user.tab(); // blur commits the text

    expect(await screen.findByText(/Sizing at 16,384 tok context/)).toBeInTheDocument();

    const useButtons = await screen.findAllByRole('button', { name: 'Use' });
    expect(useButtons.length).toBeGreaterThan(0);
    await user.click(useButtons[0]);

    // "Use" must load the same clamped context the row (and the note) promised, not the raw
    // 131,072 still sitting in the "Context per user" field.
    const patch = openInCalculator.mock.calls[0][0] as OpenInCalculatorPatch;
    expect(patch.workload?.contextTokens).toBe(16384);
    const result = calculate(patch as Parameters<typeof calculate>[0]);
    expect(result.headroomBytes).toBeGreaterThan(0);
  });
});
