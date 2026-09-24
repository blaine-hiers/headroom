import { cloudCostFor, formatNumber, formatUsd } from '../lib';
import type { CalcResult, CalcState } from '../lib';

interface Props {
  state: CalcState;
  result: CalcResult;
}

const perMillion = (v: number | undefined) => (v === undefined ? '—' : formatUsd(v));

/** Hidden unless a $/GPU-hour price is set on the Hardware panel. */
export function CostCard({ state, result }: Props) {
  const { hardware } = state;
  const concurrentUsers = state.workload.concurrentUsers;
  const cost = cloudCostFor(state, result);
  if (!cost) return null;

  const N = Math.floor(concurrentUsers);
  return (
    <div className="card">
      <h3>Cloud cost</h3>
      <div className="tput">
        <div>
          <p className="big num">{formatUsd(cost.costPerHour)}</p>
          <p className="muted">per hour, {formatNumber(hardware.gpuCount)} × {formatUsd(hardware.usdPerHour ?? 0)}/GPU-hr</p>
        </div>
        <div>
          <p className="big num">{perMillion(cost.atCurrentUsers)}</p>
          <p className="muted">
            per 1M output tokens at {formatNumber(N)} user{N === 1 ? '' : 's'}
          </p>
        </div>
        <div>
          <p className="big num">{perMillion(cost.atMaxUsers)}</p>
          <p className="muted">per 1M output tokens at max users (best case)</p>
        </div>
      </div>
      <p className="help">
        Typical on-demand cloud list price, prices as of 2026-09, approximate — not a quote. Set or clear it in the Hardware panel.
        {result.speculative.enabled && ' Priced at the speculative-decoding tok/s.'}
      </p>
    </div>
  );
}
