import { useCallback, useRef } from 'react';
import type { TabKey } from '../lib';

const TABS: ReadonlyArray<{ key: TabKey; label: string }> = [
  { key: 'calculator', label: 'Calculator' },
  { key: 'planner', label: 'Planner' },
];

interface Props {
  active: TabKey;
  onChange: (tab: TabKey) => void;
}

/**
 * The tab bar under the header (issue #21): `role="tablist"` with one `role="tab"` button per
 * tab, ARIA-wired to the `role="tabpanel"` each renders in App.tsx (`aria-controls`/`id`
 * match). Left/Right/Home/End move both focus and selection, per the standard ARIA tabs pattern
 * — only the active tab is in the Tab order (`tabIndex`), the arrow keys move between tabs.
 */
export function TabBar({ active, onChange }: Props) {
  const buttonRefs = useRef(new Map<TabKey, HTMLButtonElement>());

  const focusTab = useCallback((key: TabKey) => {
    buttonRefs.current.get(key)?.focus();
  }, []);

  const onKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex = -1;
    if (e.key === 'ArrowRight') nextIndex = (index + 1) % TABS.length;
    else if (e.key === 'ArrowLeft') nextIndex = (index - 1 + TABS.length) % TABS.length;
    else if (e.key === 'Home') nextIndex = 0;
    else if (e.key === 'End') nextIndex = TABS.length - 1;
    if (nextIndex === -1) return;
    e.preventDefault();
    const next = TABS[nextIndex];
    onChange(next.key);
    focusTab(next.key);
  };

  return (
    <div className="tabbar" role="tablist" aria-label="Sections">
      {TABS.map((t, i) => (
        <button
          key={t.key}
          ref={(el) => {
            if (el) buttonRefs.current.set(t.key, el);
            else buttonRefs.current.delete(t.key);
          }}
          type="button"
          role="tab"
          id={`tab-${t.key}`}
          aria-selected={active === t.key}
          aria-controls={`tabpanel-${t.key}`}
          tabIndex={active === t.key ? 0 : -1}
          className={`tab${active === t.key ? ' tab-active' : ''}`}
          onClick={() => onChange(t.key)}
          onKeyDown={(e) => onKeyDown(e, i)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
