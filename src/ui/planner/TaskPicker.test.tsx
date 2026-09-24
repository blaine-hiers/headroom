import { useReducer } from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { plannerReducer } from './plannerState';
import { TaskPicker } from './TaskPicker';

function Harness({ openInCalculator }: { openInCalculator: (patch: unknown) => void }) {
  const [planner, dispatch] = useReducer(plannerReducer, {});
  return <TaskPicker planner={planner} dispatch={dispatch} openInCalculator={openInCalculator as never} />;
}

describe('TaskPicker', () => {
  it('ranks the catalog for a task/hardware pick and "Use" loads that model into the Calculator on the chosen GPU', async () => {
    const user = userEvent.setup();
    const openInCalculator = vi.fn();
    render(<Harness openInCalculator={openInCalculator} />);

    await user.selectOptions(screen.getByLabelText('Task'), 'coding');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'RTX 4090');

    const rows = screen.getAllByRole('row').slice(1); // drop the header row
    expect(rows.length).toBeGreaterThan(0);

    const firstRow = rows[0];
    await user.click(within(firstRow).getByRole('button', { name: 'Use' }));

    expect(openInCalculator).toHaveBeenCalledTimes(1);
    const patch = openInCalculator.mock.calls[0][0] as { hardware: { gpuName: string; gpuCount: number }; model: { id: string } };
    expect(patch.hardware.gpuName).toBe('RTX 4090');
    expect(patch.hardware.gpuCount).toBe(1);
    expect(patch.model.id).toBeTruthy();
  });

  it('"Size hardware" hands the model off via setHandoffModelId', async () => {
    const user = userEvent.setup();

    function HandoffHarness() {
      const [planner, dispatch] = useReducer(plannerReducer, {});
      return (
        <>
          <TaskPicker planner={planner} dispatch={dispatch} openInCalculator={vi.fn()} />
          <p data-testid="handoff">{planner.handoffModelId ?? 'none'}</p>
        </>
      );
    }
    render(<HandoffHarness />);

    const rows = screen.getAllByRole('row').slice(1);
    await user.click(within(rows[0]).getByRole('button', { name: 'Size hardware' }));

    expect(screen.getByTestId('handoff').textContent).not.toBe('none');
  });

  it('shows a hint and no table when nothing in the catalog matches the filters', async () => {
    const user = userEvent.setup();
    render(<Harness openInCalculator={vi.fn()} />);

    await user.selectOptions(screen.getByLabelText('Task'), 'small-edge');
    const contextInput = screen.getByLabelText('Context needed');
    await user.clear(contextInput);
    await user.type(contextInput, '200000');
    contextInput.blur();

    expect(await screen.findByText(/No bundled model tagged/)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('states the ranking rule and that it is a heuristic, not a benchmark', () => {
    render(<Harness openInCalculator={vi.fn()} />);
    expect(screen.getByText(/Heuristic, not a benchmark/)).toBeInTheDocument();
  });
});
