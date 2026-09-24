import type { ReactNode } from 'react';
import {
  attentionParamsPerLayer,
  DECODE_EFFICIENCY,
  effectiveSlidingLayers,
  effectiveVramGB,
  findGpuPreset,
  formatNumber,
  kvContextForRuntime,
  KV_QUANTS,
  kvBytesPerTokenPerLayer,
  llamaCppPerSlotContext,
  RUNTIME_PROFILES,
  SGLANG_MEM_FRACTION_STATIC,
  VLLM_GPU_MEMORY_UTILIZATION,
  VLLM_KV_BLOCK_TOKENS,
  VLLM_OVERHEAD_ALLOWANCE_GB,
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
  const { model, quant, hardware: hw, workload, runtime } = state;
  const kvB = KV_QUANTS[quant.kv].bytesPerElement;
  const bits = WEIGHT_QUANTS[quant.weight].bitsPerWeight;
  const perLayer = kvBytesPerTokenPerLayer(model, quant.kv);
  const sliding = effectiveSlidingLayers(model);
  const full = model.numLayers - sliding;
  const C = Math.floor(workload.contextTokens);
  const N = Math.floor(workload.concurrentUsers);
  const kvCtx = kvContextForRuntime(C, runtime);
  const fixed = result.weightBytes + result.overheadBytes;
  const active = result.activeParams;
  const method = result.activeParamsMethod;
  const activeBytes = weightBytes(active, quant.weight);
  const B = (v: number) => <Bytes value={v} />;
  const maxU = result.maxUsersAtContext;

  const gpu = findGpuPreset(hw.gpuName);
  const isAppleGpu = gpu?.vendor === 'apple';
  const effectiveVram = effectiveVramGB(hw.gpuName, hw.vramGB, hw.appleWiredLimitGB);
  const appleNote = isAppleGpu && hw.vramGB !== effectiveVram ? ` (using wired limit ${n(effectiveVram, 2)} GB)` : '';

  const usableFormula =
    runtime === 'vllm'
      ? `gpuCount × effectiveVramGB × 1e9 × gpu_memory_utilization (${VLLM_GPU_MEMORY_UTILIZATION})`
      : runtime === 'sglang'
        ? `gpuCount × effectiveVramGB × 1e9 × mem_fraction_static (${SGLANG_MEM_FRACTION_STATIC})`
        : isAppleGpu
          ? 'gpuCount × effectiveVramGB × 1e9 × (1 − reserve%/100)'
          : 'gpuCount × vramGB × 1e9 × (1 − reserve%/100)';
  const usableSub =
    runtime === 'vllm'
      ? `${n(hw.gpuCount)} × ${n(effectiveVram, 2)} × 1e9 × ${VLLM_GPU_MEMORY_UTILIZATION}`
      : runtime === 'sglang'
        ? `${n(hw.gpuCount)} × ${n(effectiveVram, 2)} × 1e9 × ${SGLANG_MEM_FRACTION_STATIC}`
        : `${n(hw.gpuCount)} × ${n(effectiveVram, 2)} × 1e9 × (1 − ${n(hw.reservePct, 2)}/100)`;

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
        {runtime === 'vllm' && kvCtx !== C && (
          <Step
            title="Context rounded to vLLM's paged-KV block size"
            formula={`ceil(C / ${VLLM_KV_BLOCK_TOKENS}) × ${VLLM_KV_BLOCK_TOKENS}`}
            sub={`ceil(${n(C)} / ${VLLM_KV_BLOCK_TOKENS}) × ${VLLM_KV_BLOCK_TOKENS}`}
            result={`${n(kvCtx)} tokens`}
          />
        )}
        {sliding > 0 ? (
          <Step
            title="KV per request (sliding-window split)"
            formula="perLayer × (fullLayers × C + slidingLayers × min(C, window))"
            sub={`${n(perLayer)} × (${n(full)} × ${n(kvCtx)} + ${n(sliding)} × ${n(Math.min(kvCtx, model.slidingWindow ?? 0))})`}
            result={B(result.kvBytesPerRequest)}
          />
        ) : (
          <Step
            title="KV per request"
            formula="KV per token × C"
            sub={`${n(result.kvBytesPerToken)} × ${n(kvCtx)}`}
            result={B(result.kvBytesPerRequest)}
          />
        )}
        {runtime === 'llamacpp' && (
          <Step
            title="Per-slot context (llama.cpp: -c shared across -np slots)"
            formula="floor((C × N) / N)"
            sub={`floor((${n(C)} × ${n(N)}) / ${n(N)})`}
            result={`${n(llamaCppPerSlotContext(C * N, N))} tokens${llamaCppPerSlotContext(C * N, N) < C ? ' — below the chosen context' : ''}`}
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
          title={`Usable VRAM (${RUNTIME_PROFILES[runtime].label})${appleNote}`}
          formula={usableFormula}
          sub={usableSub}
          result={B(result.usableBytes)}
        />
        <Step
          title="Total VRAM"
          formula={
            runtime === 'vllm'
              ? `weights + (overheadGB × 1e9 × gpuCount + ${VLLM_OVERHEAD_ALLOWANCE_GB} × 1e9 × gpuCount) + N × KV per request`
              : 'weights + overheadGB × 1e9 × gpuCount + N × KV per request'
          }
          sub={
            runtime === 'vllm' ? (
              <>
                {formatNumber(result.weightBytes)} + ({n(hw.overheadGB, 2)} × 1e9 × {n(hw.gpuCount)} + {VLLM_OVERHEAD_ALLOWANCE_GB} × 1e9 × {n(hw.gpuCount)}) + {n(N)}{' '}
                × {n(result.kvBytesPerRequest)}
              </>
            ) : (
              <>
                {formatNumber(result.weightBytes)} + {n(hw.overheadGB, 2)} × 1e9 × {n(hw.gpuCount)} + {n(N)} × {n(result.kvBytesPerRequest)}
              </>
            )
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
          formula={
            runtime === 'vllm'
              ? `min(maxPosition, ${VLLM_KV_BLOCK_TOKENS} × floor((usable − fixed) / (N × KV per token) / ${VLLM_KV_BLOCK_TOKENS}))`
              : 'min(maxPosition, floor((usable − fixed) / (N × KV per token)))'
          }
          sub={
            runtime === 'vllm'
              ? `min(${n(model.maxPositionEmbeddings)}, ${VLLM_KV_BLOCK_TOKENS} × floor((${n(result.usableBytes)} − ${n(fixed)}) / (${n(Math.max(1, N))} × ${n(result.kvBytesPerToken)}) / ${VLLM_KV_BLOCK_TOKENS}))`
              : `min(${n(model.maxPositionEmbeddings)}, floor((${n(result.usableBytes)} − ${n(fixed)}) / (${n(Math.max(1, N))} × ${n(result.kvBytesPerToken)})))`
          }
          result={`${n(result.maxContextForUsers)} tokens${runtime === 'vllm' ? ' (rounded down to a block multiple, so it always fits)' : ''}`}
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
      </ol>
    </details>
  );
}
