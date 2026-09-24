import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { findModelPreset } from '../lib';
import { ProviderPicker } from './ProviderPicker';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function setup(overrides: Partial<Parameters<typeof ProviderPicker>[0]> = {}) {
  const onSelect = vi.fn();
  const onSelectHub = vi.fn();
  render(<ProviderPicker loadedModelId="meta-llama/Llama-3.3-70B-Instruct" weightQuant="bf16" onSelect={onSelect} onSelectHub={onSelectHub} {...overrides} />);
  return { onSelect, onSelectHub };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ProviderPicker', () => {
  it('opening Qwen shows Qwen models; clicking one loads it and closes the list', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse([]))));
    const { onSelect } = setup();

    await fireEvent.click(screen.getByRole('button', { name: 'Qwen' }));
    const list = screen.getByRole('group', { name: 'Qwen models' });
    expect(within(list).getByText(/Qwen3-32B/)).toBeInTheDocument();

    fireEvent.click(within(list).getByRole('button', { name: /^Qwen3-32B/ }));
    const preset = findModelPreset('Qwen/Qwen3-32B');
    expect(onSelect).toHaveBeenCalledWith(preset);
    expect(screen.queryByRole('group', { name: 'Qwen models' })).not.toBeInTheDocument();
  });

  it('Escape closes the open list', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse([]))));
    setup();
    fireEvent.click(screen.getByRole('button', { name: 'Qwen' }));
    const list = screen.getByRole('group', { name: 'Qwen models' });
    fireEvent.keyDown(list, { key: 'Escape' });
    expect(screen.queryByRole('group', { name: 'Qwen models' })).not.toBeInTheDocument();
  });

  it('only one provider is open at a time; clicking the open one again closes it', () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse([]))));
    setup();
    fireEvent.click(screen.getByRole('button', { name: 'Qwen' }));
    expect(screen.getByRole('group', { name: 'Qwen models' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Meta' }));
    expect(screen.queryByRole('group', { name: 'Qwen models' })).not.toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Meta models' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Meta' }));
    expect(screen.queryByRole('group', { name: 'Meta models' })).not.toBeInTheDocument();
  });

  it('marks the loaded model in its list and its provider button as selected', () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse([]))));
    setup({ loadedModelId: 'meta-llama/Llama-3.3-70B-Instruct' });

    const metaButton = screen.getByRole('button', { name: 'Meta' });
    expect(metaButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Qwen' })).toHaveAttribute('aria-pressed', 'false');

    fireEvent.click(metaButton);
    const row = screen.getByRole('button', { name: /^Llama 3\.3 70B/ });
    expect(row).toHaveAttribute('aria-current', 'true');
  });

  it('is keyboard accessible: aria-expanded/aria-controls wire the trigger to its list', () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse([]))));
    setup();
    const button = screen.getByRole('button', { name: 'Qwen' });
    expect(button).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(button);
    expect(button).toHaveAttribute('aria-expanded', 'true');
    const list = screen.getByRole('group', { name: 'Qwen models' });
    expect(button.getAttribute('aria-controls')).toBe(list.id);
  });

  it('shows a Hub section when the author listing returns hits', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        expect(url).toContain('author=Qwen');
        return Promise.resolve(jsonResponse([{ id: 'Qwen/Qwen-Extra-Model', downloads: 999, gated: false }]));
      }),
    );
    const { onSelectHub } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Qwen' }));

    expect(await screen.findByText('More from Qwen on the Hub')).toBeInTheDocument();
    const hubButton = screen.getByRole('button', { name: /Qwen\/Qwen-Extra-Model/ });
    fireEvent.click(hubButton);
    expect(onSelectHub).toHaveBeenCalledWith('Qwen/Qwen-Extra-Model');
  });

  it('offline (or a failed request) shows no Hub section, silently', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('offline'))),
    );
    setup();
    fireEvent.click(screen.getByRole('button', { name: 'Qwen' }));

    // The bundled list still renders while the (failing) Hub request is in flight/rejected.
    expect(screen.getByText(/Qwen3-32B/)).toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalled());
    expect(screen.queryByText(/More from Qwen on the Hub/)).not.toBeInTheDocument();
  });
});
