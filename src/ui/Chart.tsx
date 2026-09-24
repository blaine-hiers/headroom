import { formatBytes, formatBytesBinary } from '../lib';
import type { CalcResult } from '../lib';

interface Props {
  result: CalcResult;
  users: number;
}

const W = 640;
const H = 260;
const PAD = { top: 16, right: 16, bottom: 40, left: 64 };
const CAP = 512;

function niceStep(range: number, target: number): number {
  const raw = range / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

/** Total VRAM vs concurrent users (a straight line: fixed + N × KV per request). */
export function Chart({ result, users }: Props) {
  // fixedBytes/bytesPerUser already account for CPU/RAM offload when it's on (see fit.ts).
  const fixed = result.fixedBytes;
  const kv = result.bytesPerUser;
  const usable = result.usableBytes;
  const total = (n: number) => fixed + n * kv;

  const maxU = result.maxUsersAtContext;
  const xMax = Math.max(2, Math.min(CAP, Math.max(64, Number.isFinite(maxU) ? 2 * maxU : CAP, users)));
  const yTop = Math.max(total(xMax), usable, 1) * 1.08;

  const iw = W - PAD.left - PAD.right;
  const ih = H - PAD.top - PAD.bottom;
  const x = (n: number) => PAD.left + ((n - 1) / (xMax - 1)) * iw;
  const y = (b: number) => PAD.top + ih - (Math.max(0, b) / yTop) * ih;

  const xStep = niceStep(xMax, 6);
  const xTicks: number[] = [1];
  for (let t = xStep; t <= xMax; t += xStep) xTicks.push(t);
  const yStep = niceStep(yTop / 1e9, 5) * 1e9;
  const yTicks: number[] = [];
  for (let t = 0; t <= yTop; t += yStep) yTicks.push(t);

  const samples = 24;
  const points = [...new Set(Array.from({ length: samples + 1 }, (_, i) => Math.round(1 + (i * (xMax - 1)) / samples)))];
  const path = `M${x(1)},${y(total(1))} L${x(xMax)},${y(total(xMax))}`;
  const curU = Math.min(users, xMax);
  const fitsAt = (n: number) => total(n) <= usable;
  const tip = (n: number) =>
    `${n} user${n === 1 ? '' : 's'}: ${formatBytes(total(n))} (${formatBytesBinary(total(n))}) of ${formatBytes(usable)} usable`;

  return (
    <figure className="card chart">
      <figcaption>
        <h3>VRAM vs concurrent users</h3>
        <span className="muted">at the chosen context; dashed line = usable VRAM</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Chart of total VRAM from 1 to ${xMax} concurrent users`}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
            <text className="tick" x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {formatBytes(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="tick" x={x(t)} y={H - PAD.bottom + 16} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="axis-label" x={PAD.left + iw / 2} y={H - 4} textAnchor="middle">
          concurrent users
        </text>
        <text className="axis-label" transform={`translate(12 ${PAD.top + ih / 2}) rotate(-90)`} textAnchor="middle">
          total VRAM
        </text>
        <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={PAD.top + ih} y2={PAD.top + ih} />

        <line className="usable-line" x1={PAD.left} x2={W - PAD.right} y1={y(usable)} y2={y(usable)}>
          <title>{`usable VRAM: ${formatBytes(usable)} (${formatBytesBinary(usable)})`}</title>
        </line>
        <text className="usable-label" x={W - PAD.right} y={y(usable) - 6} textAnchor="end">
          usable {formatBytes(usable)}
        </text>

        <path className="total-line" d={path} />
        {points.map((n) => (
          <circle key={n} className="hover-dot" cx={x(n)} cy={y(total(n))} r={7}>
            <title>{tip(n)}</title>
          </circle>
        ))}
        <circle className={fitsAt(curU) ? 'marker marker-ok' : 'marker marker-bad'} cx={x(curU)} cy={y(total(curU))} r={5.5}>
          <title>{`current: ${tip(curU)}`}</title>
        </circle>
      </svg>
    </figure>
  );
}
