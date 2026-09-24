import type { ReactNode } from 'react';
import {
  attentionParamsPerLayer,
  DECODE_EFFICIENCY,
  effectiveSlidingLayers,
  formatNumber,
  formatSeconds,
  KV_QUANTS,
  kvBytesPerTokenPerLayer,
  WEIGHT_QUANTS,
  weightBytes,
} from '../lib';
import type { ActiveParamsMethod, CalcResult, CalcState } from '../lib';
import { Bytes } from './Bytes';

interface Props {
  state: CalcState;
  result: CalcResult;
}

const n = (v: number, d = 0) => formatNumber(v, d);

function Step({ title, formula, sub, result }: { title: string; formula: string; sub: ReactNode; result: ReactNode }) {
  return (
    <li className="math-step">
      <h4>{title}</h4>
      <code className="formula">{formula}</code>
      <code className="sub">
        = {sub} = {result}
      </code>
    </li>
  );
}

function ActiveParamsStep({ state, active, method }: { state: CalcState; active: number; method: ActiveParamsMethod }) {
  const { model } = state;
  const moe = model.moe;
  const res = `${n(active)} (${method})`;
  if (!moe || method === 'dense') {
    return <Step title="Active params (dense)" formula="params" sub={n(model.params)} result={res} />;
  }
  if (method === 'ratio') {
    return (
      <Step
        title="Active params (MoE, ratio: can understate by a third)"
        formula="params × (perToken + shared) / (experts + shared)"
        sub={`${n(model.params)} × (${moe.expertsPerToken} + ${moe.sharedExperts}) / (${moe.numExperts} + ${moe.sharedExperts})`}
        result={res}
      />
    );
  }
  const ffn = model.ffn;
  const h = model.hiddenSize;
  const dense = Math.min(model.numLayers, ffn?.firstKDense ?? 0);
  const expert = ffn?.moeIntermediateSize ?? ffn?.intermediateSize ?? 0;
  return (
    <Step
      title="Active params (MoE, structural)"
      formula="layers × attention + denseLayers × 3 × hidden × intermediate + moeLayers × (3 × hidden × expertFfn × (perToken + shared) + hidden × experts) + vocab × hidden"
      sub={`${n(model.numLayers)} × ${n(attentionParamsPerLayer(model) ?? 0)} + ${n(dense)} × 3 × ${n(h)} × ${n(ffn?.intermediateSize ?? 0)} + ${n(model.numLayers - dense)} × (3 × ${n(h)} × ${n(expert)} × (${moe.expertsPerToken} + ${moe.sharedExperts}) + ${n(h)} × ${n(moe.numExperts)}) + ${n(model.vocabSize)} × ${n(h)}`}
      result={res}
    />
  );
}

export function ShowTheMath({ state, result }: Props) {
  const { model, quant, hardware: hw, workload } = state;
  const kvB = KV_QUANTS[quant.kv].bytesPerElement;
  const bits = WEIGHT_QUANTS[quant.weight].bitsPerWeight;
  const perLayer = kvBytesPerTokenPerLayer(model, quant.kv);
  const sliding = effectiveSlidingLayers(model);
  const full = model.numLayers - sliding;
  const C = Math.floor(workload.contextTokens);
  const N = Math.floor(workload.concurrentUsers);
  const fixed = result.weightBytes + result.overheadBytes;
  const active = result.activeParams;
  const method = result.activeParamsMethod;
  const activeBytes = weightBytes(active, quant.weight);
  const B = (v: number) => <Bytes value={v} />;
  const maxU = result.maxUsersAtContext;

  return (
    <details className="card math">
      <summary>Show the math</summary>
      <ol>
        {model.attention === 'mla' ? (
          <Step
            title="KV per token (MLA)"
            formula="layers × (kv_lora_rank + qk_rope_head_dim) × kvBytes"
            sub={`${n(model.numLayers)} × (${n(model.kvLoraRank ?? 0)} + ${n(model.qkRopeHeadDim ?? 0)}) × ${kvB}`}
            result={<>{n(result.kvBytesPerToken)} B ({B(result.kvBytesPerToken)})</>}
          />
        ) : (
          <Step
            title="KV per token"
            formula="2 × layers × kvHeads × headDim × kvBytes"
            sub={`2 × ${n(model.numLayers)} × ${n(model.numKvHeads)} × ${n(model.headDim)} × ${kvB}`}
            result={<>{n(result.kvBytesPerToken)} B ({B(result.kvBytesPerToken)})</>}
          />
        )}
        {sliding > 0 ? (
          <Step
            title="KV per request (sliding-window split)"
            formula="perLayer × (fullLayers × C + slidingLayers × min(C, window))"
            sub={`${n(perLayer)} × (${n(full)} × ${n(C)} + ${n(sliding)} × ${n(Math.min(C, model.slidingWindow ?? 0))})`}
            result={B(result.kvBytesPerRequest)}
          />
        ) : (
          <Step
            title="KV per request"
            formula="KV per token × C"
            sub={`${n(result.kvBytesPerToken)} × ${n(C)}`}
            result={B(result.kvBytesPerRequest)}
          />
        )}
        <Step title="KV for all users" formula="KV per request × N" sub={`${n(result.kvBytesPerRequest)} × ${n(N)}`} result={B(result.kvBytesAllUsers)} />
        <Step
          title="Weights"
          formula="params × bitsPerWeight / 8"
          sub={`${n(model.params)} × ${bits} / 8`}
          result={B(result.weightBytes)}
        />
        <Step
          title="Usable VRAM"
          formula="gpuCount × vramGB × 1e9 × (1 − reserve%/100)"
          sub={`${n(hw.gpuCount)} × ${n(hw.vramGB, 2)} × 1e9 × (1 − ${n(hw.reservePct, 2)}/100)`}
          result={B(result.usableBytes)}
        />
        <Step
          title="Total VRAM"
          formula="weights + overheadGB × 1e9 × gpuCount + N × KV per request"
          sub={
            <>
              {formatNumber(result.weightBytes)} + {n(hw.overheadGB, 2)} × 1e9 × {n(hw.gpuCount)} + {n(N)} × {n(result.kvBytesPerRequest)}
            </>
          }
          result={
            <>
              {B(result.totalBytes)} ({result.fits ? 'fits' : 'does not fit'} in {formatNumber(result.usableBytes)} B)
            </>
          }
        />
        <Step
          title="Max users at C"
          formula="floor((usable − fixed) / KV per request)"
          sub={`floor((${n(result.usableBytes)} − ${n(fixed)}) / ${n(result.kvBytesPerRequest)})`}
          result={Number.isFinite(maxU) ? n(maxU) : '∞'}
        />
        <Step
          title="Max context for N users"
          formula="min(maxPosition, floor((usable − fixed) / (N × KV per token)))"
          sub={`min(${n(model.maxPositionEmbeddings)}, floor((${n(result.usableBytes)} − ${n(fixed)}) / (${n(Math.max(1, N))} × ${n(result.kvBytesPerToken)})))`}
          result={`${n(result.maxContextForUsers)} tokens`}
        />
        <ActiveParamsStep state={state} active={active} method={method} />
        <Step
          title="Decode throughput"
          formula="bandwidthGBs × 1e9 × gpuCount × efficiency / (activeWeights + N × KV per request)"
          sub={
            <>
              {n(hw.bandwidthGBs)} × 1e9 × {n(hw.gpuCount)} × {DECODE_EFFICIENCY} / ({n(activeBytes)} + {n(N)} × {n(result.kvBytesPerRequest)})
            </>
          }
          result={`${n(result.throughput.perUserTokS, 1)} tok/s per user, ${n(result.throughput.aggregateTokS, 1)} tok/s aggregate`}
        />
        <Step
          title={`Prefill / time to first token (compute-bound, BF16${result.prefill.headsSource === 'hiddenSize-fallback' ? ', head count unknown → hiddenSize used' : ''})`}
          formula="[2 × activeParams × C + 2 × layers × C² × queryWidth] / (tflopsBf16 × 1e12 × gpuCount × MFU)"
          sub={`[2 × ${n(active)} × ${n(C)} + 2 × ${n(model.numLayers)} × ${n(C)}² × queryWidth] / (${n(hw.tflopsBf16, 1)} × 1e12 × ${n(hw.gpuCount)} × ${result.prefill.mfu})`}
          result={`${n(result.prefill.flops)} FLOPs → ${formatSeconds(result.prefill.ttftSeconds)}`}
        />
      </ol>
    </details>
  );
}
