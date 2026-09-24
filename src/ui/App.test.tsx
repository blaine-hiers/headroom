import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { encodeState } from '../lib';
import qwenApi from '../lib/__fixtures__/qwen2.5-7b-instruct.api.json';
import qwenConfig from '../lib/__fixtures__/qwen2.5-7b-instruct.json';
import App from './App';
import { defaultState } from './state';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function badge(): HTMLElement {
  const el = document.querySelector<HTMLElement>('.badge');
  if (!el) throw new Error('fit badge not rendered');
  return el;
}

beforeEach(() => {
  window.localStorage.clear();
  window.history.replaceState(null, '', '/');
  delete document.documentElement.dataset.theme;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('App', () => {
  it('renders the default Llama 3.3 70B preset with the golden KV figure in both units', () => {
    render(<App />);
    expect(screen.getByRole('heading', { level: 1, name: 'Headroom' })).toBeInTheDocument();
    expect(screen.getByText('Will it fit? For how many? How fast?')).toBeInTheDocument();
    // 2 × 80 × 8 × 128 × 2 = 327,680 B
    expect(screen.getAllByText('328 KB').length).toBeGreaterThan(0);
    expect(screen.getAllByText('320 KiB').length).toBeGreaterThan(0);
    expect(screen.getByText('· built-in preset')).toBeInTheDocument();
  });

  it('switching to the Llama 3.1 8B chip changes the KV per token figure', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Llama 3.1 8B' }));
    // 2 × 32 × 8 × 128 × 2 = 131,072 B
    expect(screen.getAllByText('131 KB').length).toBeGreaterThan(0);
    expect(screen.getAllByText('128 KiB').length).toBeGreaterThan(0);
    expect(screen.queryByText('320 KiB')).not.toBeInTheDocument();
  });

  it('70B BF16 does not fit on one 4090 and fits on H200s', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(badge()).toHaveTextContent('Does not fit');

    await user.selectOptions(screen.getByLabelText('GPU'), 'H200');
    // 141 GB of BF16 weights vs 141 GB × 0.95 usable: one H200 is still short.
    expect(badge()).toHaveTextContent('Does not fit');

    await user.click(screen.getByRole('button', { name: 'Increase gpu count' }));
    expect(badge()).toHaveTextContent(/^Fits$/);
  });

  it('a single H200 fits 70B once weights are FP8', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('GPU'), 'H200');
    await user.selectOptions(screen.getByLabelText('Weights'), 'fp8');
    expect(badge()).toHaveTextContent(/^Fits$/);
  });

  it('Copy link writes the encoded state to the URL and the clipboard', async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, 'writeText');
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Copy link' }));
    const encoded = encodeState(defaultState);
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0][0]).toContain(`?${encoded}`);
    expect(window.location.search).toBe(`?${encoded}`);
    expect(await screen.findByText('Copied')).toBeInTheDocument();
  });

  it('loads state from the URL', () => {
    const s = { ...defaultState, workload: { contextTokens: 32768, concurrentUsers: 4 } };
    window.history.replaceState(null, '', `/?${encodeState(s)}`);
    render(<App />);
    expect(screen.getByLabelText('Concurrent users')).toHaveValue(4);
    expect(screen.getByLabelText('Context tokens')).toHaveValue(32768);
  });

  it('theme toggle flips data-theme and persists the choice', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(document.documentElement.dataset.theme).toBe('dark');
    await user.click(screen.getByRole('button', { name: 'Switch to light theme' }));
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(window.localStorage.getItem('headroom.theme')).toBe('light');
    await user.click(screen.getByRole('button', { name: 'Switch to dark theme' }));
    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  it('fetches a model from Hugging Face (200)', async () => {
    const fetchMock = vi.fn((url: string) =>
      Promise.resolve(url.includes('/api/models/') ? jsonResponse(qwenApi) : jsonResponse(qwenConfig)),
    );
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByLabelText('Hugging Face repo id');
    await user.clear(input);
    await user.type(input, 'Qwen/Qwen2.5-7B-Instruct{Enter}');

    expect(await screen.findByText('· from Hugging Face')).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    // 2 × 28 × 4 × 128 × 2 = 57,344 B
    expect(screen.getAllByText('57.3 KB').length).toBeGreaterThan(0);
    expect(screen.getAllByText('56 KiB').length).toBeGreaterThan(0);
  });

  it('shows the fetching status, then the gated error with a preset fallback (401)', async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => {
      release = r;
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        await gate;
        return jsonResponse({ error: 'gated' }, 401);
      }),
    );
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByLabelText('Hugging Face repo id');
    await user.clear(input);
    await user.type(input, 'meta-llama/Llama-3.1-8B-Instruct');
    await user.click(screen.getByRole('button', { name: 'Fetch' }));

    const panel = screen.getByRole('region', { name: 'Model' });
    expect(within(panel).getByText(/Fetching meta-llama\/Llama-3\.1-8B-Instruct/)).toBeInTheDocument();

    await act(async () => {
      release();
    });
    expect(await within(panel).findByText(/gated or private/)).toBeInTheDocument();
    // The calculator keeps its last good spec.
    expect(screen.getAllByText('328 KB').length).toBeGreaterThan(0);

    await user.click(within(panel).getByRole('button', { name: /Use the built-in Llama 3.1 8B preset/ }));
    expect(screen.getAllByText('131 KB').length).toBeGreaterThan(0);
  });
  it('retyping Max position does not clamp the context down to the minimum on the way', async () => {
    const user = userEvent.setup();
    render(<App />);
    const ctx = screen.getByLabelText('Context tokens', { exact: true });
    await user.clear(ctx);
    await user.type(ctx, '32768');
    await user.tab();
    expect(ctx).toHaveValue(32768);

    await user.click(screen.getByText('Advanced'));
    const maxPos = screen.getByLabelText('Max position');
    await user.clear(maxPos);
    await user.type(maxPos, '131072'); // "1", "13", "131" are below the 256 minimum
    await user.tab();
    expect(maxPos).toHaveValue(131072);
    expect(ctx).toHaveValue(32768);
  });

  it('a below-minimum entry is clamped when the field loses focus', async () => {
    const user = userEvent.setup();
    render(<App />);
    const ctx = screen.getByLabelText('Context tokens', { exact: true });
    await user.clear(ctx);
    await user.type(ctx, '10');
    await user.tab();
    expect(ctx).toHaveValue(256);
  });

  it('the context slider sits on its top stop at a non-power-of-two model max', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Qwen3-32B' })); // max 40960
    const ctx = screen.getByLabelText('Context tokens', { exact: true });
    await user.clear(ctx);
    await user.type(ctx, '40960');
    await user.tab();
    const slider = screen.getByRole('slider');
    expect(slider).toHaveValue(slider.getAttribute('max'));
  });
  it('context-table rows above the model max are greyed and tagged, but still shown', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Qwen3-32B' })); // max 40960
    const rows = screen.getAllByRole('row').filter((r) => r.classList.contains('over-max'));
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent('128K');
    expect(rows[0]).toHaveTextContent('> model max');
    const row32k = screen.getAllByRole('row').find((r) => r.textContent?.startsWith('32K'));
    expect(row32k).not.toHaveClass('over-max');
  });

  it('the Weights card shows active params and how they were estimated', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByText(/70\.55B active per token \(dense, all params\)/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Qwen3-30B-A3B' }));
    expect(screen.getByText(/3\.04B active per token \(MoE, structural estimate\)/)).toBeInTheDocument();
  });

  it('tabbing through the Advanced fields leaves a preset untouched', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByText('Advanced'));
    await user.click(screen.getByLabelText('Parameters')); // shows 70.55 (rounded)
    await user.tab();
    await user.click(screen.getByLabelText('Max position'));
    await user.tab();
    expect(screen.getByText('· built-in preset')).toBeInTheDocument();
  });

  it('a fetched model appears as a recent chip and reloads without calling fetch', async () => {
    const fetchMock = vi.fn((url: string) =>
      Promise.resolve(url.includes('/api/models/') ? jsonResponse(qwenApi) : jsonResponse(qwenConfig)),
    );
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    // Fetch Qwen model
    const input = screen.getByLabelText('Hugging Face repo id');
    await user.clear(input);
    await user.type(input, 'Qwen/Qwen2.5-7B-Instruct{Enter}');

    expect(await screen.findByText('· from Hugging Face')).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalled();

    // Verify the model was stored in localStorage
    const storedRecents = JSON.parse(window.localStorage.getItem('headroom.recents') || '[]');
    expect(storedRecents).toHaveLength(1);
    expect(storedRecents[0].id).toBe('Qwen/Qwen2.5-7B-Instruct');

    // Switch to a different model (Llama 3.1 8B preset)
    await user.click(screen.getByRole('button', { name: 'Llama 3.1 8B' }));
    expect(screen.getByText('· built-in preset')).toBeInTheDocument();

    // Reset the fetch mock call count
    fetchMock.mockClear();

    // Click on the Qwen recent chip (it should be rendered in the recent models section)
    // The recent chip should now be visible; find it and click it
    const recentButtons = screen.getAllByRole('button', { hidden: false });
    const qwenButton = recentButtons.find((btn) => btn.textContent?.includes('Qwen2.5-7B-Instruct'));
    expect(qwenButton).toBeInTheDocument();
    await user.click(qwenButton!);

    // Verify the model is loaded from localStorage
    expect(await screen.findByText('· from Hugging Face')).toBeInTheDocument();

    // Verify no fetch calls were made (model was loaded from localStorage)
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
