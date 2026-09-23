import { useId } from 'react';
import { KV_QUANTS, WEIGHT_QUANTS } from '../lib';
import type { KvQuantKey, Quant, WeightQuantKey } from '../lib';

const GROUPS: Array<{ label: string; keys: WeightQuantKey[] }> = [
  { label: 'Native dtypes', keys: ['fp32', 'bf16', 'fp16', 'fp8'] },
  { label: 'GGUF (llama.cpp)', keys: ['q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'q4_0', 'iq4_xs', 'q3_k_m', 'q2_k'] },
  { label: 'Other 4-bit', keys: ['awq_gptq_4bit', 'nf4'] },
];

interface Props {
  quant: Quant;
  onChange: (patch: Partial<Quant>) => void;
}

export function QuantPanel({ quant, onChange }: Props) {
  const wId = useId();
  const kId = useId();
  const note = WEIGHT_QUANTS[quant.weight].note;
  return (
    <section className="panel" aria-labelledby="quant-h">
      <h2 id="quant-h">Quantization</h2>
      <div className="field">
        <label htmlFor={wId}>Weights</label>
        <select id={wId} value={quant.weight} onChange={(e) => onChange({ weight: e.target.value as WeightQuantKey })}>
          {GROUPS.map((g) => (
            <optgroup key={g.label} label={g.label}>
              {g.keys.map((k) => (
                <option key={k} value={k}>
                  {WEIGHT_QUANTS[k].label} — {WEIGHT_QUANTS[k].bitsPerWeight} bits/weight
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {note && <p className="help">{note}</p>}
      </div>
      <div className="field">
        <label htmlFor={kId}>KV cache</label>
        <select id={kId} value={quant.kv} onChange={(e) => onChange({ kv: e.target.value as KvQuantKey })}>
          {(Object.keys(KV_QUANTS) as KvQuantKey[]).map((k) => (
            <option key={k} value={k}>
              {KV_QUANTS[k].label} — {KV_QUANTS[k].bytesPerElement} bytes/element
            </option>
          ))}
        </select>
      </div>
    </section>
  );
}
