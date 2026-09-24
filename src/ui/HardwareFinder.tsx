import { useId, useMemo, useState } from 'react';
import { GPU_VENDOR_GROUPS, findFittingHardware, formatNumber } from '../lib';
import type { CalcState, GpuVendor, HardwareFinderRow, HardwareSpec } from '../lib';
import { Bytes } from './Bytes';

type SortKey = 'vram' | 'tokps';
type VendorFilter = GpuVendor | 'all';

interface Props {
  state: CalcState;
  onApply: (patch: Partial<HardwareSpec>) => void;
}

function sortRows(rows: HardwareFinderRow[], sort: SortKey): HardwareFinderRow[] {
  const sorted = [...rows];
  if (sort === 'tokps') {
    sorted.sort((a, b) => b.result.throughput.perUserTokS - a.result.throughput.perUserTokS);
  } else {
    sorted.sort((a, b) => a.totalVramGB - b.totalVramGB);
  }
  return sorted;
}

/** Reverse hardware search: which GPU presets and counts fit the current model, quant and workload. */
export function HardwareFinder({ state, onApply }: Props) {
  const vendorId = useId();
  const sortId = useId();
  const [vendor, setVendor] = useState<VendorFilter>('all');
  const [sort, setSort] = useState<SortKey>('vram');

  const rows = useMemo(() => findFittingHardware(state), [state]);
  const filtered = vendor === 'all' ? rows : rows.filter((r) => r.gpu.vendor === vendor);
  const sorted = useMemo(() => sortRows(filtered, sort), [filtered, sort]);

  function apply(row: HardwareFinderRow) {
    const patch: Partial<HardwareSpec> = { gpuName: row.gpu.name, gpuCount: row.gpuCount };
    // Same clearing rule the GPU select applies: a wired-memory override doesn't carry to a
    // different (or non-Apple) GPU.
    if (row.gpu.name !== state.hardware.gpuName && state.hardware.appleWiredLimitGB !== undefined) {
      patch.appleWiredLimitGB = undefined;
    }
    onApply(patch);
  }

  return (
    <details className="card">
      <summary>Which hardware fits?</summary>
      <div className="grid2">
        <div className="field">
          <label htmlFor={vendorId}>Vendor</label>
          <select id={vendorId} value={vendor} onChange={(e) => setVendor(e.target.value as VendorFilter)}>
            <option value="all">All vendors</option>
            {GPU_VENDOR_GROUPS.map((v) => (
              <option key={v.vendor} value={v.vendor}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={sortId}>Sort by</label>
          <select id={sortId} value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="vram">Total VRAM (low to high)</option>
            <option value="tokps">Decode tok/s (high to low)</option>
          </select>
        </div>
      </div>
      {sorted.length === 0 ? (
        <p className="muted">Nothing in the bundled GPU table fits at 1, 2, 4 or 8 GPUs for this model, quant and workload.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">GPU</th>
                <th scope="col">Count</th>
                <th scope="col">Total VRAM</th>
                <th scope="col">Headroom</th>
                <th scope="col">Decode tok/s</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => {
                const applied = row.gpu.name === state.hardware.gpuName && row.gpuCount === state.hardware.gpuCount;
                return (
                  <tr
                    key={row.gpu.name}
                    className={applied ? 'chosen hwfinder-row' : 'hwfinder-row'}
                    role="button"
                    tabIndex={0}
                    aria-label={`Apply ${row.gpuCount} × ${row.gpu.name}`}
                    aria-current={applied ? 'true' : undefined}
                    onClick={() => apply(row)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        apply(row);
                      }
                    }}
                  >
                    <td>
                      {row.gpu.name}
                      {applied && <span className="tag">applied</span>}
                      {row.unusual && (
                        <span className="tag" title="More than 2 consumer GPUs is an unusual multi-GPU setup.">
                          unusual
                        </span>
                      )}
                    </td>
                    <td className="num">{row.gpuCount}×</td>
                    <td className="num">{formatNumber(row.totalVramGB)} GB</td>
                    <td>
                      <Bytes value={row.result.headroomBytes} />
                    </td>
                    <td className="num">{formatNumber(row.result.throughput.perUserTokS, row.result.throughput.perUserTokS < 10 ? 1 : 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="help">
        Searches only the bundled GPU table at 1, 2, 4 and 8 GPUs, using your current reserve %, overhead and runtime. A count that fails the
        tensor-parallel split check does not count as fitting.
      </p>
    </details>
  );
}
