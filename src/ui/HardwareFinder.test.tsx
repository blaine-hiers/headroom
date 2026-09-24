import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { HardwareFinder } from './HardwareFinder';
import { defaultState } from './state';

describe('HardwareFinder', () => {
  it('lists a fitting GPU/count combination and applies it on click', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    // Default state is a 70B BF16 model on a single RTX 4090 (does not fit); two 80GB H100 SXMs do.
    render(<HardwareFinder state={defaultState} onApply={onApply} />);

    await user.click(screen.getByText('Which hardware fits?'));
    const row = screen.getByRole('button', { name: 'Apply 2 × H100 SXM' });
    await user.click(row);

    expect(onApply).toHaveBeenCalledWith({ gpuName: 'H100 SXM', gpuCount: 2 });
  });

  it('supports keyboard activation (Enter) on a row', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    render(<HardwareFinder state={defaultState} onApply={onApply} />);
    await user.click(screen.getByText('Which hardware fits?'));
    const row = screen.getByRole('button', { name: 'Apply 2 × H100 SXM' });
    row.focus();
    await user.keyboard('{Enter}');
    expect(onApply).toHaveBeenCalledWith({ gpuName: 'H100 SXM', gpuCount: 2 });
  });

  it('shows a message when nothing in the bundled table fits', async () => {
    const user = userEvent.setup();
    const hugeState = {
      ...defaultState,
      model: { ...defaultState.model, params: 20e12, activeParams: 20e12 },
    };
    render(<HardwareFinder state={hugeState} onApply={vi.fn()} />);
    await user.click(screen.getByText('Which hardware fits?'));
    expect(screen.getByText(/Nothing in the bundled GPU table fits/)).toBeInTheDocument();
  });

  it('filters rows by vendor group', async () => {
    const user = userEvent.setup();
    render(<HardwareFinder state={defaultState} onApply={vi.fn()} />);
    await user.click(screen.getByText('Which hardware fits?'));
    expect(screen.getByRole('button', { name: /H100 SXM/ })).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Vendor'), 'apple');
    expect(screen.queryByRole('button', { name: /H100 SXM/ })).not.toBeInTheDocument();
  });
});
