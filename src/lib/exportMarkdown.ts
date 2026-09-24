import { calculateCloudCost } from './cost';
import { formatBytes, formatNumber, formatSeconds, formatTokens, formatUsd } from './format';
import { resolveOffload } from './offload';
import { KV_QUANTS, WEIGHT_QUANTS } from './quant';
import type { CalcResult, CalcState } from './types';

/**
 * Fit/offload verdict, mirroring Results.tsx's `fitLevel` + offloaded override. Duplicated
 * rather than imported: lib/ stays framework-free and never depends on ui/, and this is a small,
 * stable four-way classification (see src/ui/state.ts's `fitLevel` for the UI's own copy).
 */
function verdict(state: CalcState, result: CalcResult): { label: string; detail: string } {
  const offloadEnabled = resolveOffload(state.hardware.offload).enabled;
  const offloaded = offloadEnabled && result.offload.cpuLayers > 0;
  if (offloaded) {
    return {
      label: result.offload.fitsInRam ? 'Offloaded' : 'Does not fit',
      detail: `-ngl ${formatNumber(result.offload.gpuLayers)} of ${formatNumber(state.model.numLayers)} layers on GPU, ${formatNumber(result.offload.cpuLayers)} in system RAM`,
    };
  }
  if (!result.fits) return { label: 'Does not fit', detail: `${formatBytes(-result.headroomBytes)} short` };
  const tight = result.usableBytes > 0 && result.headroomBytes / result.usableBytes < 0.1;
  return { label: tight ? 'Tight' : 'Fits', detail: `${formatBytes(result.headroomBytes)} headroom` };
}

/**
 * Escapes a value for use inside a Markdown table cell: an unescaped `|` splits the row into
 * extra columns, and a raw newline breaks the table entirely. `fileWeights.label` in particular
 * comes straight from a repo's filename, so it can contain either.
 */
function mdCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\r\n|\r|\n/g, ' ');
}

function weightsLine(state: CalcState, result: CalcResult): string {
  if (result.weightSource === 'files' && state.model.fileWeights) {
    return `${formatBytes(result.weightBytes)} (${mdCell(state.model.fileWeights.label)}, from repo files)`;
  }
  return `${formatBytes(result.weightBytes)} (${WEIGHT_QUANTS[state.quant.weight].label}, estimated)`;
}

const usersLabel = (n: number) => (Number.isFinite(n) ? formatNumber(n) : '∞');

/**
 * A Markdown summary of the current configuration and result: model, quant, hardware, workload,
 * the headline figures, the context table, and a link back to this exact configuration. Meant for
 * pasting into a GitHub issue, Reddit or Discord.
 */
export function buildMarkdownSummary(state: CalcState, result: CalcResult, shareLink: string): string {
  const { model, quant, hardware, workload } = state;
  const N = workload.concurrentUsers;
  const C = workload.contextTokens;
  const v = verdict(state, result);
  const offloadEnabled = resolveOffload(hardware.offload).enabled;

  const lines: string[] = [];
  lines.push(`## Headroom: ${model.name || model.id}`);
  lines.push('');
  lines.push(`- **Model:** ${model.id}`);
  lines.push(`- **Quant:** weight ${WEIGHT_QUANTS[quant.weight].label}, KV ${KV_QUANTS[quant.kv].label}`);
  lines.push(`- **Hardware:** ${hardware.gpuCount} × ${hardware.gpuName}`);
  lines.push(`- **Workload:** ${formatNumber(N)} user${N === 1 ? '' : 's'} @ ${formatTokens(C)} context`);
  lines.push('');
  lines.push(`**${v.label}** — ${v.detail}`);
  lines.push('');
  lines.push('| Metric | Value |');
  lines.push('|---|---|');
  lines.push(`| Weights | ${weightsLine(state, result)} |`);
  lines.push(`| KV per token | ${formatBytes(result.kvBytesPerToken)} |`);
  lines.push(`| Total VRAM | ${formatBytes(result.totalBytes)} of ${formatBytes(result.usableBytes)} usable |`);
  lines.push(`| Max users @ ${formatTokens(C)} | ${usersLabel(result.maxUsersAtContext)} |`);
  lines.push(`| Max context @ ${formatNumber(N)} user${N === 1 ? '' : 's'} | ${formatTokens(result.maxContextForUsers)} |`);
  lines.push(`| Decode tok/s | ${formatNumber(result.throughput.perUserTokS, 1)} per user, ${formatNumber(result.throughput.aggregateTokS, 1)} aggregate |`);
  lines.push(`| Time to first token | ${formatSeconds(result.prefill.ttftSeconds)} |`);

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
  if (cost) {
    const perMillion = cost.atCurrentUsers !== undefined ? `, ${formatUsd(cost.atCurrentUsers)}/1M tokens` : '';
    lines.push(`| Cloud cost | ${formatUsd(cost.costPerHour)}/hr${perMillion} |`);
  }

  if (offloadEnabled && result.offload.cpuLayers > 0) {
    lines.push(`| CPU offload | -ngl ${formatNumber(result.offload.gpuLayers)} of ${formatNumber(model.numLayers)} layers |`);
  }

  if (result.speculative.enabled) {
    lines.push(`| Speculative decoding | ×${formatNumber(result.speculative.throughput.multiplier, 2)} vs. no speculation |`);
  }

  lines.push('');
  lines.push('### Context table');
  lines.push('');
  lines.push('| Context | KV per request | Max users |');
  lines.push('|---|---|---|');
  for (const row of result.contextTable) {
    lines.push(`| ${formatTokens(row.contextTokens)} | ${formatBytes(row.kvBytesPerRequest)} | ${usersLabel(row.maxUsers)} |`);
  }

  lines.push('');
  lines.push(`[Open this configuration in Headroom](${shareLink})`);

  return lines.join('\n');
}
