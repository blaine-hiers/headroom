import { useId, useRef, useState } from 'react';
import { activeParamsDetailed, fetchModel, findModelPreset, MODEL_PRESETS } from '../lib';
import type { Attention, ModelSpec, MoeSpec, NativeDtype } from '../lib';
import { NumberField } from './NumberField';
import { readStorage, TOKEN_KEY, writeStorage } from './storage';

interface Props {
  model: ModelSpec;
  onLoad: (spec: ModelSpec) => void;
  onEdit: (patch: Partial<ModelSpec>) => void;
}

type FetchState = { kind: 'idle' } | { kind: 'fetching'; id: string } | { kind: 'error'; id: string; error: string };

const SOURCE_LABEL: Record<ModelSpec['source'], string> = {
  hf: 'from Hugging Face',
  preset: 'built-in preset',
  manual: 'manual',
};

const BIG = 1e7;

export function ModelPanel({ model, onLoad, onEdit }: Props) {
  const inputId = useId();
  const tokenId = useId();
  const [repoId, setRepoId] = useState(model.id);
  const [token, setToken] = useState(() => readStorage(TOKEN_KEY) ?? '');
  const [fetchState, setFetchState] = useState<FetchState>({ kind: 'idle' });
  const requestSeq = useRef(0);

  const doFetch = async () => {
    const id = repoId.trim();
    const seq = ++requestSeq.current;
    setFetchState({ kind: 'fetching', id });
    const res = await fetchModel(id, token || undefined);
    if (seq !== requestSeq.current) return; // a newer fetch or preset pick superseded this one
    if (res.ok) {
      setFetchState({ kind: 'idle' });
      setRepoId(res.spec.id);
      onLoad(res.spec);
    } else {
      setFetchState({ kind: 'error', id, error: res.error });
    }
  };

  const pickPreset = (spec: ModelSpec) => {
    requestSeq.current++;
    setFetchState({ kind: 'idle' });
    setRepoId(spec.id);
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
      <div className="field">
        <label htmlFor={inputId}>Hugging Face repo id</label>
        <div className="row">
          <input
            id={inputId}
            type="text"
            placeholder="org/model"
            spellCheck={false}
            autoComplete="off"
            value={repoId}
            onChange={(e) => setRepoId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                void doFetch();
              }
            }}
          />
          <button type="button" className="btn btn-primary" onClick={() => void doFetch()} disabled={fetchState.kind === 'fetching'}>
            Fetch
          </button>
        </div>
      </div>

      <div className="chips" role="group" aria-label="Built-in model presets">
        {MODEL_PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className="chip"
            aria-pressed={model.source === 'preset' && model.id === p.id}
            onClick={() => pickPreset(p)}
          >
            {p.name}
          </button>
        ))}
      </div>

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
              setToken(e.target.value);
              writeStorage(TOKEN_KEY, e.target.value.trim());
            }}
          />
          <p className="help">Sent only to huggingface.co as a Bearer token; stored only in this browser.</p>
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
