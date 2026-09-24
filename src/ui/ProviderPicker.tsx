import { useEffect, useId, useRef, useState } from 'react';
import { activeParamsDetailed, findCatalogEntry, formatBytes, formatNumber, formatTokens, listAuthorModels, PROVIDERS, providerModelRows, WEIGHT_QUANTS } from '../lib';
import type { HubSearchHit, ModelSpec, WeightQuantKey } from '../lib';

interface Props {
  /** The currently loaded model's id: marks its row and its provider's button. */
  loadedModelId: string;
  weightQuant: WeightQuantKey;
  token?: string;
  /** Loads a bundled catalog model the same way a preset chip did (adds it to recents). */
  onSelect: (spec: ModelSpec) => void;
  /** Runs the same fetch path as typing a Hub id and pressing Enter. */
  onSelectHub: (id: string) => void;
}

type HubState = { kind: 'idle' } | { kind: 'ready'; hits: HubSearchHit[] };

const fmtB = (n: number): string => `${Number((n / 1e9).toFixed(2))}B`;

/**
 * Provider row + an inline model list for whichever provider is open. Replaces the old flat
 * preset chip row: pick a provider (Meta, Qwen, ...) to see its bundled models, pick a model
 * to load it. Only one provider's list is open at a time; Esc or re-clicking the provider
 * closes it. An optional "More from <provider> on the Hub" section lists that org's
 * top-downloaded models from the Hub API, hidden whenever the request fails or is empty.
 */
export function ProviderPicker({ loadedModelId, weightQuant, token, onSelect, onSelectHub }: Props) {
  const baseId = useId();
  const [openId, setOpenId] = useState<string | undefined>(undefined);
  const [prevOpenId, setPrevOpenId] = useState(openId);
  const [hub, setHub] = useState<HubState>({ kind: 'idle' });
  const seqRef = useRef(0);
  const buttonRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const openProvider = PROVIDERS.find((p) => p.id === openId);
  const loadedProviderId = findCatalogEntry(loadedModelId)?.meta.provider;

  // Switching (or closing) providers drops the previous provider's Hub hits immediately, so a
  // stale list never flashes while the new one loads. Adjusted during render (React's documented
  // pattern for this, same as RepoSearch's value-reset) rather than in an effect, since it's
  // deriving state from a prop/state change, not synchronizing with an external system.
  if (openId !== prevOpenId) {
    setPrevOpenId(openId);
    setHub({ kind: 'idle' });
  }

  useEffect(() => {
    if (!openProvider) return;
    const seq = ++seqRef.current;
    void listAuthorModels(openProvider.hubOrg, token).then((hits) => {
      if (seq !== seqRef.current) return;
      if (hits.length > 0) setHub({ kind: 'ready', hits });
    });
  }, [openProvider, token]);

  const closeAndFocus = (id: string) => {
    setOpenId(undefined);
    buttonRefs.current[id]?.focus();
  };

  const rows = openProvider ? providerModelRows(openProvider.id, weightQuant, loadedModelId) : [];
  const bundledIds = new Set(rows.map((r) => r.entry.spec.id.toLowerCase()));
  const hubExtras = hub.kind === 'ready' ? hub.hits.filter((h) => !bundledIds.has(h.id.toLowerCase())) : [];

  return (
    <div className="provider-picker">
      <div className="chips" role="group" aria-label="Model providers">
        {PROVIDERS.map((p) => {
          const listId = `${baseId}-${p.id}-list`;
          const isOpen = openId === p.id;
          const isSelected = loadedProviderId === p.id;
          return (
            <button
              key={p.id}
              type="button"
              ref={(el) => {
                buttonRefs.current[p.id] = el;
              }}
              className={`chip provider-chip${isSelected ? ' is-selected' : ''}`}
              aria-expanded={isOpen}
              aria-controls={listId}
              aria-pressed={isSelected}
              onClick={() => setOpenId((cur) => (cur === p.id ? undefined : p.id))}
            >
              {p.name}
            </button>
          );
        })}
      </div>

      {openProvider && (
        <div
          className="provider-models"
          id={`${baseId}-${openProvider.id}-list`}
          role="group"
          aria-label={`${openProvider.name} models`}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              e.preventDefault();
              closeAndFocus(openProvider.id);
            }
          }}
        >
          <ul className="provider-model-list">
            {rows.map(({ entry, weightBytes: bytes, isLoaded }) => {
              const { spec, meta } = entry;
              const showActive = !!spec.moe && spec.moe.numExperts > 1;
              return (
                <li key={spec.id}>
                  <button
                    type="button"
                    className={`provider-model-row${isLoaded ? ' is-loaded' : ''}`}
                    aria-current={isLoaded ? 'true' : undefined}
                    onClick={() => {
                      onSelect(spec);
                      closeAndFocus(openProvider.id);
                    }}
                  >
                    <span className="provider-model-name">
                      {spec.name}
                      {isLoaded && <span className="tag"> loaded</span>}
                    </span>
                    <span className="provider-model-meta muted">
                      {` ${fmtB(spec.params)} params${showActive ? ` (${fmtB(activeParamsDetailed(spec).active)} active)` : ''} · ${formatTokens(
                        spec.maxPositionEmbeddings,
                      )} ctx · ${formatBytes(bytes)} at ${WEIGHT_QUANTS[weightQuant].label}`}
                    </span>
                    <span className="provider-model-tags">
                      {meta.tags.map((t) => (
                        <span key={t} className="tag">
                          {' '}
                          {t}
                        </span>
                      ))}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>

          {hubExtras.length > 0 && (
            <div className="provider-hub">
              <p className="help">More from {openProvider.name} on the Hub</p>
              <ul className="provider-model-list">
                {hubExtras.map((hit) => (
                  <li key={hit.id}>
                    <button
                      type="button"
                      className="provider-model-row"
                      onClick={() => {
                        onSelectHub(hit.id);
                        closeAndFocus(openProvider.id);
                      }}
                    >
                      <span className="provider-model-name">{hit.id}</span>
                      <span className="provider-model-meta muted">{formatNumber(hit.downloads)} downloads</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
