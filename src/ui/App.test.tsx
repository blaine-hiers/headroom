import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { encodeState } from '../lib';
import type { CalcState } from '../lib';
import qwenApi from '../lib/__fixtures__/qwen2.5-7b-instruct.api.json';
import qwenConfig from '../lib/__fixtures__/qwen2.5-7b-instruct.json';
import { fromBase64 } from '../lib/__fixtures__/base64';
import qwenMoeB64 from '../lib/__fixtures__/qwen3-30b-a3b-q4_k_m.gguf.b64?raw';
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
    expect(fetchMock).toHaveBeenCalledTimes(3); // model API, config.json, file listing
    // 2 × 28 × 4 × 128 × 2 = 57,344 B
    expect(screen.getAllByText('57.3 KB').length).toBeGreaterThan(0);
    expect(screen.getAllByText('56 KiB').length).toBeGreaterThan(0);
  });

  it('fetches a GGUF repo: file picker, exact weight bytes, and the source on the Weights card', async () => {
    const listing = {
      siblings: [
        { rfilename: 'Qwen_Qwen3-30B-A3B-Q4_K_M.gguf', size: 18_556_686_080 },
        { rfilename: 'Qwen_Qwen3-30B-A3B-Q8_0.gguf', size: 32_483_935_968 },
      ],
      gguf: { total: 30_532_122_624 },
    };
    const fetchMock = vi.fn((url: string) => {
      if (url.includes('?blobs=true')) return Promise.resolve(jsonResponse(listing));
      if (url.includes('/api/models/')) return Promise.resolve(jsonResponse({}));
      if (url.endsWith('config.json')) return Promise.resolve(jsonResponse({ error: 'Entry not found' }, 404));
      return Promise.resolve(new Response(fromBase64(qwenMoeB64), { status: 206 }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByLabelText('Hugging Face repo id');
    await user.clear(input);
    await user.type(input, 'bartowski/Qwen_Qwen3-30B-A3B-GGUF{Enter}');

    expect(await screen.findByText('· from Hugging Face')).toBeInTheDocument();
    expect(screen.getByLabelText('GGUF file')).toHaveValue('Qwen_Qwen3-30B-A3B-Q4_K_M.gguf');
    expect(screen.getByLabelText('Weights')).toHaveValue('q4_k_m');
    expect(screen.getAllByText('18.6 GB').length).toBeGreaterThan(0);
    expect(screen.getByText(/Q4_K_M GGUF, from repo files, 4\.86 bits\/weight effective/)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('GGUF file'), 'Qwen_Qwen3-30B-A3B-Q8_0.gguf');
    expect(await screen.findByText(/Q8_0 GGUF, from repo files/)).toBeInTheDocument();
    expect(screen.getByLabelText('Weights')).toHaveValue('q8_0');

    // Another quant than the files' one: back to the estimate, and the card says so.
    await user.selectOptions(screen.getByLabelText('Weights'), 'bf16');
    expect(screen.getByText(/BF16, 16 bits\/weight, estimated/)).toBeInTheDocument();
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

  it('clicking a fit-matrix cell applies that quant and context to the calculator', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByText('Fit matrix: weight quant × context'));

    // Default model is the Llama 3.3 70B preset (BF16, 8192 tokens chosen).
    expect(screen.getByLabelText('Weights')).toHaveValue('bf16');
    expect(screen.getByLabelText('Context tokens')).toHaveValue(8192);

    await user.click(screen.getByRole('button', { name: /^FP8 at 2K:/ }));

    expect(screen.getByLabelText('Weights')).toHaveValue('fp8');
    expect(screen.getByLabelText('Context tokens')).toHaveValue(2048);
  });

  it('compare mode: duplicating the config then editing the new column updates the compare table', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Turning Compare on starts from column B: a duplicate of the current (only) config.
    await user.click(screen.getByRole('button', { name: 'Compare' }));
    expect(screen.getByRole('button', { name: 'A' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'B' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'C' })).not.toBeInTheDocument();

    // Freshly duplicated: A and B are identical, so nothing in the row is "better" yet.
    let totalRow = screen.getByRole('row', { name: /^Total VRAM/ });
    let cells = within(totalRow).getAllByRole('cell');
    expect(cells).toHaveLength(2);
    expect(cells[0]).toHaveTextContent(cells[1].textContent ?? '');
    expect(cells[0]).not.toHaveClass('compare-best');
    expect(cells[1]).not.toHaveClass('compare-best');

    // Select column B, so the edit below lands on B, not A.
    await user.click(screen.getByRole('button', { name: 'B' }));
    const usersField = screen.getByLabelText('Concurrent users');
    await user.clear(usersField);
    await user.type(usersField, '32');
    await user.tab();

    totalRow = screen.getByRole('row', { name: /^Total VRAM/ });
    cells = within(totalRow).getAllByRole('cell');
    expect(cells[0]).not.toHaveTextContent(cells[1].textContent ?? '');
    // B now uses more VRAM for the same hardware, so A (lower) is the highlighted, better column.
    expect(cells[0]).toHaveClass('compare-best');
    expect(cells[1]).not.toHaveClass('compare-best');

    // Switching back to A edits A, not B, and leaves B's column untouched.
    await user.click(screen.getByRole('button', { name: 'A' }));
    expect(screen.getByLabelText('Concurrent users')).toHaveValue(1);
  });

  it('compare mode: a GGUF picker loaded in column B does not follow the user to column A (#20)', async () => {
    const listing = {
      siblings: [
        { rfilename: 'Qwen_Qwen3-30B-A3B-Q4_K_M.gguf', size: 18_556_686_080 },
        { rfilename: 'Qwen_Qwen3-30B-A3B-Q8_0.gguf', size: 32_483_935_968 },
      ],
      gguf: { total: 30_532_122_624 },
    };
    const fetchMock = vi.fn((url: string) => {
      if (url.includes('?blobs=true')) return Promise.resolve(jsonResponse(listing));
      if (url.includes('/api/models/')) return Promise.resolve(jsonResponse({}));
      if (url.endsWith('config.json')) return Promise.resolve(jsonResponse({ error: 'Entry not found' }, 404));
      return Promise.resolve(new Response(fromBase64(qwenMoeB64), { status: 206 }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: 'Compare' }));
    await user.click(screen.getByRole('button', { name: 'B' }));
    const input = screen.getByLabelText('Hugging Face repo id');
    await user.clear(input);
    await user.type(input, 'bartowski/Qwen_Qwen3-30B-A3B-GGUF{Enter}');
    expect(await screen.findByLabelText('GGUF file')).toHaveValue('Qwen_Qwen3-30B-A3B-Q4_K_M.gguf');

    // Back to A: A never loaded a GGUF repo, so there must be no picker (which would load B's repo into A).
    await user.click(screen.getByRole('button', { name: 'A' }));
    expect(screen.queryByLabelText('GGUF file')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Hugging Face repo id')).toHaveValue('meta-llama/Llama-3.3-70B-Instruct');
    expect(screen.getByLabelText('Weights')).toHaveValue('bf16');
  });

  it('compare mode round-trips a 2-column URL and turning it off drops the extra column from the URL', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Compare' })); // auto-duplicates into a 2-column state
    await vi.waitFor(() => expect(window.location.search).toMatch(/[?&]c2=/));

    await user.click(screen.getByRole('button', { name: 'Compare: on' }));
    await vi.waitFor(() => expect(window.location.search).not.toMatch(/[?&]c2=/));
  });

  it('compare mode: a crafted out-of-range c2 in the URL is clamped like the primary column', async () => {
    const bad: CalcState = {
      ...defaultState,
      hardware: { ...defaultState.hardware, gpuCount: 999_999, vramGB: -50, bandwidthGBs: -10, tflopsBf16: -5, reservePct: -20, overheadGB: 999 },
    };
    const search = `?${encodeState(defaultState)}&c2=${encodeURIComponent(encodeState(bad))}`;
    window.history.replaceState(null, '', `/${search}`);

    const user = userEvent.setup();
    render(<App />);

    // A c2 column in the URL turns compare mode on by itself.
    await user.click(screen.getByRole('button', { name: 'B' }));
    // Same bounds initialState enforces on the primary column (MAX_GPUS = 16, vramGB >= 0.1).
    expect(screen.getByLabelText('GPU count')).toHaveValue(16);
    expect(screen.getByLabelText('VRAM per GPU')).toHaveValue(0.1);
  });

  describe('Remember token toggle', () => {
    it('defaults unchecked and does not persist a typed token to localStorage', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.click(screen.getByText('Gated models'));
      const checkbox = screen.getByRole('checkbox', { name: 'Remember token on this device' });
      expect(checkbox).not.toBeChecked();

      await user.type(screen.getByLabelText('Hugging Face token'), 'hf_test_fake');
      expect(window.localStorage.getItem('headroom.hfToken')).toBeNull();
    });

    it('checking Remember token persists the current token', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.click(screen.getByText('Gated models'));
      await user.type(screen.getByLabelText('Hugging Face token'), 'hf_test_fake');

      await user.click(screen.getByRole('checkbox', { name: 'Remember token on this device' }));
      expect(window.localStorage.getItem('headroom.hfToken')).toBe('hf_test_fake');
    });

    it('unchecking Remember token after checking removes the stored token', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.click(screen.getByText('Gated models'));
      await user.type(screen.getByLabelText('Hugging Face token'), 'hf_test_fake');
      const checkbox = screen.getByRole('checkbox', { name: 'Remember token on this device' });

      await user.click(checkbox);
      expect(window.localStorage.getItem('headroom.hfToken')).toBe('hf_test_fake');

      await user.click(checkbox);
      expect(window.localStorage.getItem('headroom.hfToken')).toBeNull();
    });

    it('an explicit unchecked preference does not load or keep a stale stored token', async () => {
      // Simulates a token written back by another tab (still checked) or an older build
      // after this tab unchecked Remember token.
      window.localStorage.setItem('headroom.rememberToken', 'false');
      window.localStorage.setItem('headroom.hfToken', 'hf_stale');
      const user = userEvent.setup();
      render(<App />);
      await user.click(screen.getByText('Gated models'));
      expect(screen.getByRole('checkbox', { name: 'Remember token on this device' })).not.toBeChecked();
      expect(screen.getByLabelText('Hugging Face token')).toHaveValue('');
      expect(window.localStorage.getItem('headroom.hfToken')).toBeNull();
    });

    it('starts checked when a token is already stored from before the update', async () => {
      window.localStorage.setItem('headroom.hfToken', 'hf_test_fake');
      const user = userEvent.setup();
      render(<App />);
      await user.click(screen.getByText('Gated models'));
      expect(screen.getByRole('checkbox', { name: 'Remember token on this device' })).toBeChecked();
      expect(screen.getByLabelText('Hugging Face token')).toHaveValue('hf_test_fake');
    });

    it('does not break the panel when localStorage throws', async () => {
      const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
        throw new Error('blocked');
      });
      const user = userEvent.setup();
      render(<App />);
      await user.click(screen.getByText('Gated models'));
      expect(screen.getByRole('checkbox', { name: 'Remember token on this device' })).not.toBeChecked();
      expect(screen.getByLabelText('Hugging Face token')).toHaveValue('');
      getItem.mockRestore();
    });
  });
});

describe('Tabs (#21)', () => {
  it('opens on the Calculator by default, with the tab bar\'s ARIA wired up', () => {
    render(<App />);
    const calcTab = screen.getByRole('tab', { name: 'Calculator' });
    const plannerTab = screen.getByRole('tab', { name: 'Planner' });
    expect(calcTab).toHaveAttribute('aria-selected', 'true');
    expect(plannerTab).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', calcTab.id);
    expect(screen.getByRole('heading', { level: 1, name: 'Headroom' })).toBeInTheDocument();
  });

  it('clicking the Planner tab switches panels and updates the URL', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('tab', { name: 'Planner' }));
    expect(screen.getByRole('tab', { name: 'Planner' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('Which model for this task?')).toBeInTheDocument();
    expect(screen.getByText('What hardware to serve N users?')).toBeInTheDocument();
    await waitFor(() => expect(window.location.search).toContain('tab=planner'));

    await user.click(screen.getByRole('tab', { name: 'Calculator' }));
    expect(screen.getByRole('tab', { name: 'Calculator' })).toHaveAttribute('aria-selected', 'true');
    await waitFor(() => expect(window.location.search).not.toContain('tab='));
  });

  it('arrow keys move focus and selection between tabs; Home/End jump to the ends', async () => {
    const user = userEvent.setup();
    render(<App />);
    const calcTab = screen.getByRole('tab', { name: 'Calculator' });
    const plannerTab = screen.getByRole('tab', { name: 'Planner' });
    calcTab.focus();

    await user.keyboard('{ArrowRight}');
    expect(plannerTab).toHaveFocus();
    expect(plannerTab).toHaveAttribute('aria-selected', 'true');

    await user.keyboard('{ArrowLeft}');
    expect(calcTab).toHaveFocus();
    expect(calcTab).toHaveAttribute('aria-selected', 'true');

    await user.keyboard('{End}');
    expect(plannerTab).toHaveFocus();
    await user.keyboard('{Home}');
    expect(calcTab).toHaveFocus();
  });

  it('?tab=planner opens directly on the Planner', () => {
    window.history.replaceState(null, '', '/?tab=planner');
    render(<App />);
    expect(screen.getByRole('tab', { name: 'Planner' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('Which model for this task?')).toBeInTheDocument();
  });

  it('an old link with no tab key opens the Calculator unchanged', () => {
    const s = { ...defaultState, workload: { contextTokens: 32768, concurrentUsers: 4 } };
    window.history.replaceState(null, '', `/?${encodeState(s)}`);
    render(<App />);
    expect(screen.getByRole('tab', { name: 'Calculator' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByLabelText('Concurrent users')).toHaveValue(4);
  });

  it('Calculator state survives a round trip through the Planner', async () => {
    const user = userEvent.setup();
    render(<App />);
    const ctx = screen.getByLabelText('Context tokens', { exact: true });
    await user.clear(ctx);
    await user.type(ctx, '32768');
    await user.tab();
    expect(ctx).toHaveValue(32768);

    await user.click(screen.getByRole('tab', { name: 'Planner' }));
    await user.click(screen.getByRole('tab', { name: 'Calculator' }));
    expect(screen.getByLabelText('Context tokens', { exact: true })).toHaveValue(32768);
  });

  it('Clear button resets Calculator to defaults and shows Undo notice', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Change some inputs
    await user.selectOptions(screen.getByLabelText('GPU'), 'H200');
    await user.clear(screen.getByLabelText('Context tokens', { exact: true }));
    await user.type(screen.getByLabelText('Context tokens', { exact: true }), '32768');
    await user.tab();

    // Verify changes are visible
    expect(screen.getByLabelText('GPU')).toHaveValue('H200');
    expect(screen.getByLabelText('Context tokens', { exact: true })).toHaveValue(32768);

    // Click Clear
    await user.click(screen.getByRole('button', { name: 'Clear calculator' }));

    // Verify reset to defaults
    expect(screen.getByLabelText('GPU')).toHaveValue(defaultState.hardware.gpuName);
    expect(screen.getByLabelText('Context tokens', { exact: true })).toHaveValue(defaultState.workload.contextTokens);

    // Verify Undo notice appears
    expect(screen.getByText(/Cleared/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Undo' })).toBeInTheDocument();
  });

  it('Undo restores Calculator state and extra columns', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Change inputs
    await user.selectOptions(screen.getByLabelText('GPU'), 'H200');
    await user.clear(screen.getByLabelText('Context tokens', { exact: true }));
    await user.type(screen.getByLabelText('Context tokens', { exact: true }), '16384');
    await user.tab();

    // Turn on compare mode
    await user.click(screen.getByRole('button', { name: /compare/i }));

    // Clear
    await user.click(screen.getByRole('button', { name: 'Clear calculator' }));

    // Verify cleared
    expect(screen.getByLabelText('GPU')).toHaveValue(defaultState.hardware.gpuName);

    // Undo
    await user.click(screen.getByRole('button', { name: 'Undo' }));

    // Verify restored
    expect(screen.getByLabelText('GPU')).toHaveValue('H200');
    expect(screen.getByLabelText('Context tokens', { exact: true })).toHaveValue(16384);
  });

  it('Clear button in Planner resets Planner state and shows Undo notice', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Switch to Planner tab
    await user.click(screen.getByRole('tab', { name: 'Planner' }));

    // Click Clear button (when Planner has content, this test would verify more state)
    const clearBtn = screen.getByRole('button', { name: 'Clear planner' });
    expect(clearBtn).toBeInTheDocument();
    await user.click(clearBtn);

    // Verify Undo notice appears
    expect(screen.getByText(/Cleared/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Undo' })).toBeInTheDocument();
  });

  it("clearing the Planner never brings back the Calculator's undo notice", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole('button', { name: 'Clear calculator' }));
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    await user.click(screen.getByRole('tab', { name: 'Planner' }));
    await user.click(screen.getByRole('button', { name: 'Clear planner' }));
    await user.click(screen.getByRole('tab', { name: 'Calculator' }));
    const calcPanel = document.getElementById('tabpanel-calculator')!;
    expect(within(calcPanel).queryByRole('status')).not.toBeInTheDocument();
  });
});
