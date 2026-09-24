import { calculateCloudCost, formatNumber, formatUsd } from '../lib';
import type { CalcResult, HardwareSpec } from '../lib';

interface Props {
  hardware: HardwareSpec;
  result: CalcResult;
  concurrentUsers: number;
}

const perMillion = (v: number | undefined) => (v === undefined ? '—' : formatUsd(v));

/** Hidden unless a $/GPU-hour price is set on the Hardware panel. */
export function CostCard({ hardware, result, concurrentUsers }: Props) {
  const cost = calculateCloudCost({
    usdPerHour: hardware.usdPerHour,
    gpuCount: hardware.gpuCount,
    bandwidthGBs: hardware.bandwidthGBs,
    activeWeightBytes: result.activeWeightBytes,
    kvBytesPerRequest: result.kvBytesPerRequest,
    efficiency: result.throughput.efficiency,
    aggregateTokS: result.throughput.aggregateTokS,
    maxUsersAtContext: result.maxUsersAtContext,
  });
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
      <p className="help">Typical on-demand cloud list price, prices as of 2026-09, approximate — not a quote. Set or clear it in the Hardware panel.</p>
    </div>
  );
}
