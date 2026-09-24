import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useReducer } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { calculate } from '../../lib';
import type { OpenInCalculatorPatch } from '../state';
import { HardwareSizing } from './HardwareSizing';
import { initialPlannerState, plannerReducer } from './plannerState';

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
});
