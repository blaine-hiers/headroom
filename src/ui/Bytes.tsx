import { formatBytes, formatBytesBinary } from '../lib';

interface Props {
  value: number;
  /** Stack the binary figure under the decimal one (for large headline numbers). */
  stacked?: boolean;
}

/**
 * Decimal bytes as the primary figure (GPU vendors quote GB), binary as muted
 * secondary text (most LLM material quotes GiB): "328 KB · 320 KiB".
 */
export function Bytes({ value, stacked = false }: Props) {
  const dec = formatBytes(value);
  const bin = formatBytesBinary(value);
  return (
    <span className={stacked ? 'bytes bytes-stacked' : 'bytes'} title={`${dec} = ${bin}`}>
      <span className="bytes-dec num">{dec}</span>
      {bin !== dec && <span className="bytes-bin num">{bin}</span>}
    </span>
  );
}
