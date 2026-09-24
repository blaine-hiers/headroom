import { useId, useRef, useState } from 'react';
import { activeParamsDetailed, fetchRepo, findModelPreset, formatBytes, MODEL_PRESETS } from '../lib';
import type { Attention, GgufOption, ModelSpec, MoeSpec, NativeDtype, WeightQuantKey } from '../lib';
import { NumberField } from './NumberField';
import { ProviderPicker } from './ProviderPicker';
import { RepoSearch } from './RepoSearch';
import { readStorage, REMEMBER_TOKEN_KEY, TOKEN_KEY, writeStorage } from './storage';
import { getRecents, addRecent, removeRecent, clearRecents } from './recents';

interface Props {
  model: ModelSpec;
  weightQuant: WeightQuantKey;
  onLoad: (spec: ModelSpec) => void;
  onEdit: (patch: Partial<ModelSpec>) => void;
}

type FetchState = { kind: 'idle'; note?: string } | { kind: 'fetching'; id: string } | { kind: 'error'; id: string; error: string };

const SOURCE_LABEL: Record<ModelSpec['source'], string> = {
  hf: 'from Hugging Face',
  preset: 'built-in preset',
  manual: 'manual',
};

const BIG = 1e7;

export function ModelPanel({ model, weightQuant, onLoad, onEdit }: Props) {
  const inputId = useId();
  const tokenId = useId();
  const rememberId = useId();
  // Resolve remember-vs-token together, once, so an explicit "don't remember" preference
  // never leaves a stale token (written by another tab, or an older build) loaded into
  // state or sitting in storage.
  const [tokenInit] = useState(() => {
    const pref = readStorage(REMEMBER_TOKEN_KEY);
    // No preference saved yet: an existing stored token means an existing user, who
    // should start checked so their saved token keeps working. New users default unchecked.
    const remember = pref !== null ? pref === 'true' : readStorage(TOKEN_KEY) !== null;
    if (!remember) {
      if (pref === 'false') writeStorage(TOKEN_KEY, null);
      return { token: '', remember };
    }
    return { token: readStorage(TOKEN_KEY) ?? '', remember };
  });
  const [token, setToken] = useState(tokenInit.token);
  const [rememberToken, setRememberToken] = useState(tokenInit.remember);
  const [fetchState, setFetchState] = useState<FetchState>({ kind: 'idle' });
  const [recents, setRecents] = useState(() => getRecents());
  const [gguf, setGguf] = useState<{ id: string; options: GgufOption[]; selected: string } | undefined>(undefined);
  const requestSeq = useRef(0);

  const doFetch = async (repo: string, ggufPath?: string) => {
    const id = repo.trim();
    const seq = ++requestSeq.current;
    setFetchState({ kind: 'fetching', id });
    const res = await fetchRepo(id, token || undefined, ggufPath);
    if (seq !== requestSeq.current) return; // a newer fetch or preset pick superseded this one
    if (res.ok) {
      setFetchState(res.note ? { kind: 'idle', note: res.note } : { kind: 'idle' });
      addRecent(res.spec);
      setRecents(getRecents());
      setGguf(res.gguf && { id: res.spec.id, ...res.gguf });
      onLoad(res.spec);
    } else {
      setFetchState({ kind: 'error', id, error: res.error });
    }
  };

  const pickPreset = (spec: ModelSpec) => {
    requestSeq.current++;
    setFetchState({ kind: 'idle' });
    setGguf(undefined);
    addRecent(spec);
    setRecents(getRecents());
    onLoad(spec);
  };

  const fallback = fetchState.kind === 'error' ? findModelPreset(fetchState.id) : undefined;
  const moe = model.moe;
  const setMoe = (patch: Partial<MoeSpec>) => {
    if (!moe) return;
    onEdit({ moe: { ...moe, ...patch } });
  };

  return (
    <section className="panel" aria-labelledby="model-h">
      <h2 id="model-h">Model</h2>
      <RepoSearch
        value={model.id}
        presets={MODEL_PRESETS}
        token={token || undefined}
        fetching={fetchState.kind === 'fetching'}
        onSubmit={(id) => void doFetch(id)}
        onSelectPreset={pickPreset}
        onSelectHub={(id) => void doFetch(id)}
      />

      {gguf && (
        <div className="field">
          <label htmlFor={`${inputId}-gguf`}>GGUF file</label>
          <select
            id={`${inputId}-gguf`}
            value={gguf.selected}
            disabled={fetchState.kind === 'fetching'}
            onChange={(e) => void doFetch(gguf.id, e.target.value)}
          >
            {gguf.options.map((o) => (
              <option key={o.path} value={o.path} title={o.path}>
                {o.label} · {formatBytes(o.bytes)}
                {o.shards > 1 ? ` · ${o.shards} parts` : ''}
              </option>
            ))}
          </select>
        </div>
      )}

      <ProviderPicker loadedModelId={model.id} weightQuant={weightQuant} token={token || undefined} onSelect={pickPreset} onSelectHub={(id) => void doFetch(id)} />

      {recents.length > 0 && (
        <div className="chips" role="group" aria-label="Recent models">
          {recents.map((r) => (
            <div key={r.id} className="chip-with-remove">
              <button
                type="button"
                className="chip"
                aria-pressed={model.id === r.id}
                onClick={() => pickPreset(r)}
              >
                {r.name}
              </button>
              <button
                type="button"
                className="chip-remove"
                aria-label={`Remove ${r.name} from recent`}
                onClick={() => {
                  removeRecent(r.id);
                  setRecents(getRecents());
                }}
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className="link-btn"
            onClick={() => {
              clearRecents();
              setRecents([]);
            }}
          >
            Clear recent
          </button>
        </div>
      )}

      <p className={`status status-${fetchState.kind}`} aria-live="polite" role="status">
        {fetchState.kind === 'fetching' && <>Fetching {fetchState.id}…</>}
        {fetchState.kind === 'error' && (
          <>
            Error: {fetchState.error}
            {fallback && (
              <>
                {' '}
                <button type="button" className="link-btn" onClick={() => pickPreset(fallback)}>
                  Use the built-in {fallback.name} preset
                </button>
              </>
            )}
          </>
        )}
        {fetchState.kind === 'idle' && (
          <>
            <strong>{model.name}</strong> <span className="muted">· {SOURCE_LABEL[model.source]}</span>
            {fetchState.note && <span className="muted"> · {fetchState.note}</span>}
          </>
        )}
      </p>

      <details className="disclosure">
        <summary>Gated models</summary>
        <div className="field">
          <label htmlFor={tokenId}>Hugging Face token</label>
          <input
            id={tokenId}
            type="password"
            autoComplete="off"
            spellCheck={false}
            placeholder="hf_…"
            value={token}
            onChange={(e) => {
              const next = e.target.value;
              setToken(next);
              if (rememberToken) writeStorage(TOKEN_KEY, next.trim());
            }}
          />
          <label className="check">
            <input
              id={rememberId}
              type="checkbox"
              checked={rememberToken}
              onChange={(e) => {
                const checked = e.target.checked;
                setRememberToken(checked);
                writeStorage(REMEMBER_TOKEN_KEY, checked ? 'true' : 'false');
                writeStorage(TOKEN_KEY, checked ? token.trim() || null : null);
              }}
            />
            Remember token on this device
          </label>
          <p className="help">
            {rememberToken
              ? 'Sent only to huggingface.co as a Bearer token; stored only in this browser.'
              : 'Sent only to huggingface.co as a Bearer token; kept only until you reload.'}
          </p>
        </div>
      </details>

      <details className="disclosure">
        <summary>Advanced</summary>
        <div className="grid2">
          <NumberField
            label="Parameters"
            suffix="B"
            value={model.params / 1e9}
            decimals={2}
            step={0.01}
            min={0}
            max={100000}
            onChange={(v) => onEdit({ params: Math.round(v * 1e9) })}
          />
          <NumberField label="Layers" value={model.numLayers} integer min={1} max={1000} onChange={(v) => onEdit({ numLayers: v })} />
          <div className="field">
            <label htmlFor={`${inputId}-att`}>Attention</label>
            <select
              id={`${inputId}-att`}
              value={model.attention}
              onChange={(e) => {
                const attention = e.target.value as Attention;
                onEdit(
                  attention === 'mla'
                    ? { attention, kvLoraRank: model.kvLoraRank ?? 512, qkRopeHeadDim: model.qkRopeHeadDim ?? 64 }
                    : { attention },
                );
              }}
            >
              <option value="mha_gqa">MHA / GQA</option>
              <option value="mla">MLA (latent)</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${inputId}-dt`}>Native dtype</label>
            <select id={`${inputId}-dt`} value={model.nativeDtype} onChange={(e) => onEdit({ nativeDtype: e.target.value as NativeDtype })}>
              <option value="bf16">BF16</option>
              <option value="fp16">FP16</option>
              <option value="fp32">FP32</option>
              <option value="fp8">FP8</option>
            </select>
          </div>
          {model.attention === 'mla' ? (
            <>
              <NumberField label="KV LoRA rank" value={model.kvLoraRank ?? 0} integer min={0} max={BIG} onChange={(v) => onEdit({ kvLoraRank: v })} />
              <NumberField label="QK RoPE head dim" value={model.qkRopeHeadDim ?? 0} integer min={0} max={BIG} onChange={(v) => onEdit({ qkRopeHeadDim: v })} />
            </>
          ) : (
            <>
              <NumberField label="KV heads" value={model.numKvHeads} integer min={1} max={BIG} onChange={(v) => onEdit({ numKvHeads: v })} />
              <NumberField label="Head dim" value={model.headDim} integer min={1} max={BIG} onChange={(v) => onEdit({ headDim: v })} />
            </>
          )}
          <NumberField
            label="Sliding window"
            suffix="tok"
            value={model.slidingWindow ?? 0}
            integer
            min={0}
            max={BIG}
            onChange={(v) => onEdit({ slidingWindow: v })}
          />
          <NumberField
            label="Sliding layers"
            value={model.slidingLayers ?? 0}
            integer
            min={0}
            max={model.numLayers}
            onChange={(v) => onEdit({ slidingLayers: v })}
          />
          <NumberField
            label="Max position"
            suffix="tok"
            value={model.maxPositionEmbeddings}
            integer
            min={256}
            max={BIG * 10}
            commitOnBlur
            onChange={(v) => onEdit({ maxPositionEmbeddings: v })}
          />
          <NumberField label="Hidden size" value={model.hiddenSize} integer min={0} max={BIG} onChange={(v) => onEdit({ hiddenSize: v })} />
          <NumberField label="Vocab size" value={model.vocabSize} integer min={0} max={BIG} onChange={(v) => onEdit({ vocabSize: v })} />
        </div>

        <label className="check">
          <input
            type="checkbox"
            checked={!!moe}
            onChange={(e) =>
              onEdit({ moe: e.target.checked ? { numExperts: 8, expertsPerToken: 2, sharedExperts: 0 } : undefined })
            }
          />
          Mixture of experts
        </label>
        {moe && (
          <div className="grid3">
            <NumberField label="Experts" value={moe.numExperts} integer min={1} max={100000} onChange={(v) => setMoe({ numExperts: v })} />
            <NumberField label="Per token" value={moe.expertsPerToken} integer min={1} max={100000} onChange={(v) => setMoe({ expertsPerToken: v })} />
            <NumberField label="Shared" value={moe.sharedExperts} integer min={0} max={100000} onChange={(v) => setMoe({ sharedExperts: v })} />
          </div>
        )}
        <p className="help">
          Active params: <span className="num">{(model.activeParams / 1e9).toFixed(2)} B</span>
          {` (${activeParamsDetailed(model).method})`}. Any edit marks the model as manual.
        </p>
      </details>

      {model.warnings.length > 0 && (
        <ul className="warnings" aria-label="Model notes">
          {model.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
