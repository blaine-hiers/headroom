import { useEffect, useId, useRef, useState } from 'react';
import { formatNumber, fuzzyMatches, searchHub } from '../lib';
import type { HubSearchHit, ModelSpec } from '../lib';

const DEBOUNCE_MS = 250;
const MAX_PRESET_MATCHES = 5;

type SearchItem = { kind: 'preset'; spec: ModelSpec } | { kind: 'hub'; hit: HubSearchHit };

interface Props {
  /** The currently loaded model's id. The field is a search box, not a display of the loaded
   * model (ModelPanel's "Loaded:" line is): it starts empty, and any change to this id (preset
   * pick, a successful fetch, a Planner hand-off) clears it again. Typing does not feed back
   * through this prop. */
  loadedId: string;
  presets: ModelSpec[];
  token?: string;
  fetching: boolean;
  onSubmit: (id: string) => void;
  onSelectPreset: (spec: ModelSpec) => void;
  onSelectHub: (id: string) => void;
}

/**
 * The repo id field: a combobox that fuzzy-matches bundled presets locally (so it works
 * offline) and, ~250ms after typing stops, searches the Hub for matching repos. Selecting
 * an entry (click or Enter) runs the same fetch path as typing an id and pressing Enter.
 */
export function RepoSearch({ loadedId, presets, token, fetching, onSubmit, onSelectPreset, onSelectHub }: Props) {
  const inputId = useId();
  const listId = useId();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [hubHits, setHubHits] = useState<HubSearchHit[]>([]);
  const [prevLoadedId, setPrevLoadedId] = useState(loadedId);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const seqRef = useRef(0);

  const cancelPendingSearch = () => {
    clearTimeout(timerRef.current);
    seqRef.current++; // invalidates any in-flight response, even one already in flight
  };

  // An external load (preset pick, successful fetch) clears the field; the user's own typing
  // never comes back through `loadedId`, so this never fires mid-keystroke. Adjusted during
  // render (React's documented pattern for this) rather than in an effect, so it takes
  // effect in the same commit instead of triggering an extra render.
  if (loadedId !== prevLoadedId) {
    setPrevLoadedId(loadedId);
    setQuery('');
    setOpen(false);
    setActiveIndex(-1);
    setHubHits([]);
  }

  // Refs (the debounce timer and its sequence number) aren't render state, so they're
  // cancelled from an effect instead of the render-phase branch above: a stray in-flight
  // search for the text that was just replaced should never repopulate the list.
  useEffect(() => {
    cancelPendingSearch();
    return () => clearTimeout(timerRef.current);
  }, [loadedId]);

  const q = query.trim();
  const presetMatches = q ? presets.filter((p) => fuzzyMatches(q, p.id, p.name)).slice(0, MAX_PRESET_MATCHES) : [];
  const presetIds = new Set(presetMatches.map((p) => p.id));
  const hubMatches = q ? hubHits.filter((h) => !presetIds.has(h.id)) : [];
  const items: SearchItem[] = [
    ...presetMatches.map((spec): SearchItem => ({ kind: 'preset', spec })),
    ...hubMatches.map((hit): SearchItem => ({ kind: 'hub', hit })),
  ];
  const showList = open && items.length > 0;

  const selectItem = (item: SearchItem) => {
    cancelPendingSearch();
    setOpen(false);
    setActiveIndex(-1);
    if (item.kind === 'preset') onSelectPreset(item.spec);
    else onSelectHub(item.hit.id);
  };

  const handleChange = (v: string) => {
    setQuery(v);
    setActiveIndex(-1);
    setOpen(true);
    cancelPendingSearch();
    const nextQuery = v.trim();
    if (!nextQuery) {
      setHubHits([]);
      return;
    }
    const seq = seqRef.current;
    timerRef.current = setTimeout(() => {
      void searchHub(nextQuery, token).then((hits) => {
        if (seq === seqRef.current) setHubHits(hits);
      });
    }, DEBOUNCE_MS);
  };

  const submit = () => {
    cancelPendingSearch();
    setOpen(false);
    onSubmit(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (items.length === 0) return;
      setOpen(true);
      setActiveIndex((i) => (i + 1) % items.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (items.length === 0) return;
      setOpen(true);
      setActiveIndex((i) => (i - 1 + items.length) % items.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (open && activeIndex >= 0 && activeIndex < items.length) {
        selectItem(items[activeIndex]);
      } else {
        submit();
      }
    } else if (e.key === 'Escape') {
      if (open) {
        e.preventDefault();
        setOpen(false);
        setActiveIndex(-1);
      }
    }
  };

  const activeId = activeIndex >= 0 && activeIndex < items.length ? `${listId}-opt-${activeIndex}` : undefined;

  return (
    <div className="field">
      <label htmlFor={inputId}>Hugging Face repo id</label>
      <div className="row">
        <div className="combobox-wrap">
          <input
            id={inputId}
            type="text"
            role="combobox"
            aria-expanded={showList}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={activeId}
            placeholder="Search or paste a repo id, e.g. Qwen/Qwen3-8B"
            spellCheck={false}
            autoComplete="off"
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            onFocus={() => {
              if (q) setOpen(true);
            }}
            onBlur={() => {
              cancelPendingSearch();
              setOpen(false);
              setActiveIndex(-1);
            }}
            onKeyDown={handleKeyDown}
          />
          {showList && (
            <ul className="combobox-list" id={listId} role="listbox" onMouseDown={(e) => e.preventDefault()}>
              {items.map((item, i) => {
                const key = item.kind === 'preset' ? `preset-${item.spec.id}` : `hub-${item.hit.id}`;
                const optId = `${listId}-opt-${i}`;
                const repoId = item.kind === 'preset' ? item.spec.id : item.hit.id;
                return (
                  <li
                    key={key}
                    id={optId}
                    role="option"
                    aria-selected={i === activeIndex}
                    className={`combobox-option${i === activeIndex ? ' is-active' : ''}`}
                    onClick={() => selectItem(item)}
                  >
                    <span className="combobox-id">{repoId}</span>
                    {item.kind === 'preset' ? (
                      <span className="muted"> · built-in preset</span>
                    ) : (
                      <>
                        <span className="muted"> · {formatNumber(item.hit.downloads)} downloads</span>
                        {item.hit.gated && <span className="combobox-gated">gated</span>}
                      </>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <button type="button" className="btn btn-primary" onClick={submit} disabled={fetching}>
          Fetch
        </button>
      </div>
    </div>
  );
}
