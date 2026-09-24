import { effectiveBitsPerWeight, formatNumber, formatTokens, KV_QUANTS, WEIGHT_QUANTS } from '../lib';
import type { ActiveParamsMethod, CalcResult, CalcState } from '../lib';
import { Bytes } from './Bytes';
import { Chart } from './Chart';
import { ShowTheMath } from './ShowTheMath';
import { fitLevel } from './state';

interface Props {
  state: CalcState;
  result: CalcResult;
}

const BADGE_TEXT = { fits: 'Fits', tight: 'Tight', nofit: 'Does not fit' } as const;

const METHOD_LABEL: Record<ActiveParamsMethod, string> = {
  dense: 'dense, all params',
  structural: 'MoE, structural estimate',
  ratio: 'MoE, ratio estimate',
};

/** 3.04e9 → "3.04B". */
const billions = (n: number) => `${formatNumber(n / 1e9, 2)}B`;

const users = (n: number) => (Number.isFinite(n) ? formatNumber(n) : '∞');
const tokS = (v: number) => (v > 0 && Number.isFinite(v) ? formatNumber(v, v < 10 ? 1 : 0) : '—');

/** "1, 2, 4 or 8" */
const listWithOr = (nums: number[]) =>
  nums.length <= 1 ? (nums[0]?.toString() ?? '') : `${nums.slice(0, -1).join(', ')} or ${nums[nums.length - 1]}`;

function tensorParallelWarnings(tp: CalcResult['tensorParallel'], numKvHeads: number): string[] {
  const msgs: string[] = [];
  // Shared across both messages: suggestions already satisfy the head-count AND KV-head checks.
  const suggestion = tp.suggestedGpuCounts.length > 0 ? `; use ${listWithOr(tp.suggestedGpuCounts)}` : '';
  if (tp.checkable && !tp.headsDivisible) {
    msgs.push(`${tp.gpuCount} GPUs can't evenly split ${formatNumber(tp.numAttentionHeads ?? 0)} attention heads for tensor parallelism${suggestion}`);
  }
  if (tp.kvHeadsReplicated && !tp.kvHeadsSplitValid) {
    msgs.push(`${tp.gpuCount} GPUs can't evenly split or replicate ${formatNumber(numKvHeads)} KV heads for tensor parallelism${suggestion}`);
  } else if (tp.kvHeadsReplicated) {
    // Not MLA (kvHeadsReplicated is always false there): fewer KV heads than GPUs still lays
    // out evenly, but every GPU holding a full replicated head raises the per-GPU KV size.
    msgs.push(`Fewer KV heads than GPUs: KV heads are replicated, raising per-GPU KV size ×${formatNumber(tp.kvReplicationFactor, 2)}.`);
  }
  return msgs;
}

export function Results({ state, result }: Props) {
  const { workload, quant, hardware, model } = state;
  const N = workload.concurrentUsers;
  const C = workload.contextTokens;
  const level = fitLevel(result.fits, result.headroomBytes, result.usableBytes);
  const fixedTooBig = result.weightBytes + result.overheadBytes > result.usableBytes;
  const tpWarnings = tensorParallelWarnings(result.tensorParallel, model.numKvHeads);

  return (
    <div className="results">
      <section className={`card verdict verdict-${level}`} aria-labelledby="verdict-h">
        <h2 id="verdict-h" className="visually-hidden">
          Verdict
        </h2>
        <div className="badge" aria-live="polite" role="status">
          {BADGE_TEXT[level]}
        </div>
        <p className="verdict-detail">
          {result.fits ? (
            <>
              <Bytes value={result.headroomBytes} /> headroom
            </>
          ) : (
            <>
              <Bytes value={-result.headroomBytes} /> short
            </>
          )}
          <span className="muted">
            {' '}
            · {hardware.gpuCount} × {hardware.gpuName}, {formatNumber(N)} user{N === 1 ? '' : 's'} @ {formatTokens(C)}
          </span>
        </p>
        {tpWarnings.length > 0 && (
          <ul className="warnings" aria-label="Tensor-parallel notes">
            {tpWarnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </section>

      <div className="cards">
        <div className="card stat">
          <h3>KV per token</h3>
          <Bytes value={result.kvBytesPerToken} stacked />
          <p className="muted">{KV_QUANTS[quant.kv].label} cache, all layers</p>
        </div>
        <div className="card stat">
          <h3>Weights</h3>
          <Bytes value={result.weightBytes} stacked />
          {result.weightSource === 'files' ? (
            <p className="muted weight-source">
              {model.fileWeights?.label}, from repo files
              {model.params > 0 && <>, {formatNumber(effectiveBitsPerWeight(result.weightBytes, model.params), 2)} bits/weight effective</>}
            </p>
          ) : (
            <p className="muted weight-source">
              {WEIGHT_QUANTS[quant.weight].label}, {WEIGHT_QUANTS[quant.weight].bitsPerWeight} bits/weight, estimated
            </p>
          )}
          <p className="muted active-params">
            {billions(result.activeParams)} active per token ({METHOD_LABEL[result.activeParamsMethod]})
          </p>
        </div>
        <div className="card stat">
          <h3>
            KV for {formatNumber(N)} user{N === 1 ? '' : 's'} @ {formatTokens(C)}
          </h3>
          <Bytes value={result.kvBytesAllUsers} stacked />
          <p className="muted">
            <Bytes value={result.kvBytesPerRequest} /> per request
          </p>
        </div>
        <div className="card stat">
          <h3>Total VRAM</h3>
          <Bytes value={result.totalBytes} stacked />
          <p className="muted">
            of <Bytes value={result.usableBytes} /> usable
          </p>
        </div>
      </div>

      <div className="cards cards-2">
        <div className="card callout">
          <h3>Max users at {formatTokens(C)}</h3>
          <p className="big num">{users(result.maxUsersAtContext)}</p>
          <p className="muted">{fixedTooBig ? 'weights + overhead alone exceed usable VRAM' : 'concurrent requests, each at full context'}</p>
        </div>
        <div className="card callout">
          <h3>
            Max context for {formatNumber(N)} user{N === 1 ? '' : 's'}
          </h3>
          <p className="big num">{formatTokens(result.maxContextForUsers)}</p>
          <p className="muted">
            {formatNumber(result.maxContextForUsers)} tokens (full-attention rate, capped at the model max)
          </p>
        </div>
      </div>

      <div className="card">
        <h3>Context table</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Context</th>
                <th scope="col">KV per request</th>
                <th scope="col">Max users</th>
              </tr>
            </thead>
            <tbody>
              {result.contextTable.map((row) => {
                const chosen = row.contextTokens === Math.floor(C);
                // Past the model's max position: still shown so the 128K comparison stays visible.
                const overMax = row.contextTokens > model.maxPositionEmbeddings;
                const cls = [chosen && 'chosen', overMax && 'over-max'].filter(Boolean).join(' ');
                return (
                  <tr key={row.contextTokens} className={cls || undefined} aria-current={chosen ? 'true' : undefined}>
                    <td className="num">
                      {formatTokens(row.contextTokens)}
                      {chosen && <span className="tag">chosen</span>}
                      {overMax && <span className="tag">&gt; model max</span>}
                    </td>
                    <td>
                      <Bytes value={row.kvBytesPerRequest} />
                    </td>
                    <td className="num">{users(row.maxUsers)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Decode throughput</h3>
        <div className="tput">
          <div>
            <p className="big num">{tokS(result.throughput.perUserTokS)}</p>
            <p className="muted">tok/s per user</p>
          </div>
          <div>
            <p className="big num">{tokS(result.throughput.aggregateTokS)}</p>
            <p className="muted">tok/s aggregate</p>
          </div>
        </div>
        <p className="help">
          bandwidth-bound decode estimate, ×{formatNumber(result.throughput.efficiency, 3)} efficiency; prefill not included
          {hardware.gpuCount > 1 && ' (includes an estimated tensor-parallel communication penalty; see Show the math)'}
        </p>
      </div>

      <Chart result={result} users={N} />
      <ShowTheMath state={state} result={result} />
      <p className="caveat muted">Estimates, not benchmarks. Real runtimes add activation memory, fragmentation, and their own KV block rounding.</p>
    </div>
  );
}
