import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ModelSpec } from '../lib';
import { RepoSearch } from './RepoSearch';

const PRESETS: ModelSpec[] = [
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct',
    name: 'Llama 3.1 8B',
    params: 8e9,
    activeParams: 8e9,
    numLayers: 32,
    attention: 'mha_gqa',
    numKvHeads: 8,
    headDim: 128,
    maxPositionEmbeddings: 131072,
    hiddenSize: 4096,
    vocabSize: 128256,
    nativeDtype: 'bf16',
    source: 'preset',
    warnings: [],
  },
];

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function setup(overrides: Partial<Parameters<typeof RepoSearch>[0]> = {}) {
  const onSubmit = vi.fn();
  const onSelectPreset = vi.fn();
  const onSelectHub = vi.fn();
  render(
    <RepoSearch
      loadedId=""
      presets={PRESETS}
      fetching={false}
      onSubmit={onSubmit}
      onSelectPreset={onSelectPreset}
      onSelectHub={onSelectHub}
      {...overrides}
    />,
  );
  const input = screen.getByLabelText('Hugging Face repo id');
  return { input, onSubmit, onSelectPreset, onSelectHub };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('RepoSearch', () => {
  it('matches a bundled preset locally, with no network call, as soon as it is typed', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const { input } = setup();
    fireEvent.change(input, { target: { value: 'llama' } });
    expect(screen.getByRole('option', { name: /meta-llama\/Llama-3\.1-8B-Instruct/ })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('debounces the Hub search by ~250ms and shows results with downloads and a gated badge', async () => {
    const fetchMock = vi.fn((_url: string) =>
      Promise.resolve(
        jsonResponse([{ id: 'meta-llama/Llama-3.2-1B-Instruct', downloads: 7388679, gated: 'manual' }]),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);
    const { input } = setup();

    fireEvent.change(input, { target: { value: 'llama-3.2' } });
    expect(fetchMock).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(249);
    });
    expect(fetchMock).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain('search=llama-3.2');

    const option = screen.getByRole('option', { name: /meta-llama\/Llama-3\.2-1B-Instruct/ });
    expect(option).toHaveTextContent('7,388,679 downloads');
    expect(option).toHaveTextContent('gated');
  });

  it('only fires one search for rapid keystrokes (each keystroke resets the debounce)', async () => {
    const fetchMock = vi.fn((_url: string) => Promise.resolve(jsonResponse([])));
    vi.stubGlobal('fetch', fetchMock);
    const { input } = setup();

    fireEvent.change(input, { target: { value: 'll' } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });
    fireEvent.change(input, { target: { value: 'llama' } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(249);
    });
    expect(fetchMock).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain('search=llama');
  });

  it('shows no hub options when the search returns empty', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse([]))));
    const { input } = setup();
    fireEvent.change(input, { target: { value: 'zzznomatch' } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('stays silent and keeps working when the search request fails (offline / rate limit)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('offline'))),
    );
    const { input } = setup();
    fireEvent.change(input, { target: { value: 'llama' } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });
    // The preset match still renders; nothing threw.
    expect(screen.getByRole('option', { name: /meta-llama\/Llama-3\.1-8B-Instruct/ })).toBeInTheDocument();
  });

  it('supports full keyboard navigation: arrows move, Enter selects, Escape closes', async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(jsonResponse([{ id: 'org/hub-match', downloads: 42, gated: false }])),
    );
    vi.stubGlobal('fetch', fetchMock);
    const { input, onSelectPreset, onSelectHub } = setup();

    fireEvent.change(input, { target: { value: 'llama' } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    const listbox = screen.getByRole('listbox');
    expect(input).toHaveAttribute('aria-expanded', 'true');
    expect(input).toHaveAttribute('aria-controls', listbox.id);

    // First arrow-down highlights the first option (the bundled preset).
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    const options = screen.getAllByRole('option');
    expect(options[0]).toHaveAttribute('aria-selected', 'true');
    expect(input).toHaveAttribute('aria-activedescendant', options[0].id);

    // Second arrow-down moves to the hub result.
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(options[1]).toHaveAttribute('aria-selected', 'true');

    // Escape closes the list without selecting.
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    expect(onSelectPreset).not.toHaveBeenCalled();
    expect(onSelectHub).not.toHaveBeenCalled();

    // Re-open and select the hub entry with Enter.
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSelectHub).toHaveBeenCalledWith('org/hub-match');
    expect(onSelectPreset).not.toHaveBeenCalled();
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('Enter with no highlighted suggestion submits the typed text', () => {
    vi.stubGlobal('fetch', vi.fn());
    const { input, onSubmit } = setup();
    fireEvent.change(input, { target: { value: 'org/model' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).toHaveBeenCalledWith('org/model');
  });

  it('clicking an option selects it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(jsonResponse([]))),
    );
    const { input, onSelectPreset } = setup();
    fireEvent.change(input, { target: { value: 'llama' } });
    const option = screen.getByRole('option', { name: /meta-llama\/Llama-3\.1-8B-Instruct/ });
    fireEvent.click(option);
    expect(onSelectPreset).toHaveBeenCalledWith(PRESETS[0]);
  });

  it('starts empty with a placeholder, even when a model is already loaded (#28)', () => {
    render(
      <RepoSearch
        loadedId="meta-llama/Llama-3.3-70B-Instruct"
        presets={PRESETS}
        fetching={false}
        onSubmit={vi.fn()}
        onSelectPreset={vi.fn()}
        onSelectHub={vi.fn()}
      />,
    );
    const input = screen.getByLabelText('Hugging Face repo id');
    expect(input).toHaveValue('');
    expect(input).toHaveAttribute('placeholder', expect.stringMatching(/Search or paste a repo id/));
  });

  it('an external load (a new loaded id) clears the field and closes the list', () => {
    const onSubmit = vi.fn();
    const onSelectPreset = vi.fn();
    const onSelectHub = vi.fn();
    const { rerender } = render(
      <RepoSearch
        loadedId=""
        presets={PRESETS}
        fetching={false}
        onSubmit={onSubmit}
        onSelectPreset={onSelectPreset}
        onSelectHub={onSelectHub}
      />,
    );
    const input = screen.getByLabelText('Hugging Face repo id');
    fireEvent.change(input, { target: { value: 'llama' } });
    expect(screen.getByRole('listbox')).toBeInTheDocument();

    rerender(
      <RepoSearch
        loadedId="meta-llama/Llama-3.1-8B-Instruct"
        presets={PRESETS}
        fetching={false}
        onSubmit={onSubmit}
        onSelectPreset={onSelectPreset}
        onSelectHub={onSelectHub}
      />,
    );
    expect(input).toHaveValue('');
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('re-loading the id that is already loaded still clears the field (loadSeq bump)', () => {
    const props = {
      loadedId: 'meta-llama/Llama-3.1-8B-Instruct',
      presets: PRESETS,
      fetching: false,
      onSubmit: vi.fn(),
      onSelectPreset: vi.fn(),
      onSelectHub: vi.fn(),
    };
    const { rerender } = render(<RepoSearch {...props} loadSeq={1} />);
    const input = screen.getByLabelText('Hugging Face repo id');
    fireEvent.change(input, { target: { value: 'Llama-3.1-8B' } });
    expect(input).toHaveValue('Llama-3.1-8B');

    rerender(<RepoSearch {...props} loadSeq={2} />);
    expect(input).toHaveValue('');
  });
});
