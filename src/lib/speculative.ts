import { kvBytesForContext } from './kvcache';
import { decodeThroughput } from './throughput';
import type { KvQuantKey, SpeculativeConfig, SpeculativeMemory, SpeculativeThroughput } from './types';
import { activeParamsDetailed, weightBytes } from './weights';

export const DEFAULT_DRAFT_K = 4;
export const DEFAULT_DRAFT_ALPHA = 0.7;

/** Off by default: no draft model, no memory or speed impact. */
export const DISABLED_SPECULATIVE: SpeculativeConfig = {
  enabled: false,
  draftMode: 'none',
  draftWeightQuant: 'q4_k_m',
  k: DEFAULT_DRAFT_K,
  alpha: DEFAULT_DRAFT_ALPHA,
};

/**
 * Expected tokens produced per verify step: sum_{i=0}^{k} α^i = (1 − α^(k+1)) / (1 − α).
 * α = 1 is a removable 0/0 singularity — every draft token is always accepted, so the limit
 * there is k + 1 tokens (the k drafts plus the target's own bonus token).
 */
export function expectedTokensPerStep(alpha: number, k: number): number {
  const kk = Math.max(0, Math.floor(k));
  if (alpha >= 1) return kk + 1;
  if (alpha <= 0) return 1;
  return (1 - alpha ** (kk + 1)) / (1 - alpha);
}

function draftWeights(cfg: SpeculativeConfig): { fullWeightBytes: number; activeWeightBytes: number } {
  if (cfg.draftMode === 'preset' && cfg.draftModel) {
    const full = weightBytes(cfg.draftModel.params, cfg.draftWeightQuant);
    const active = weightBytes(activeParamsDetailed(cfg.draftModel).active, cfg.draftWeightQuant);
    return { fullWeightBytes: full, activeWeightBytes: active };
  }
  if (cfg.draftMode === 'custom') {
    // No known architecture: treated as dense (all params read every step).
    const full = weightBytes(cfg.draftParams ?? 0, cfg.draftWeightQuant);
    return { fullWeightBytes: full, activeWeightBytes: full };
  }
  return { fullWeightBytes: 0, activeWeightBytes: 0 };
}

/**
 * Draft model memory: its weights, plus its own KV cache at the same context and user count
 * as the target. A 'custom' (params-only) draft has no known architecture, so only 'preset'
 * (which carries a full ModelSpec) contributes a KV estimate; 'none' (n-gram) costs nothing.
 */
export function speculativeMemory(
  cfg: SpeculativeConfig,
  contextTokens: number,
  concurrentUsers: number,
  kvQuant: KvQuantKey,
): SpeculativeMemory {
  if (!cfg.enabled) {
    return { weightBytes: 0, activeWeightBytes: 0, kvBytesPerRequest: 0, kvBytesAllUsers: 0, totalBytes: 0 };
  }
  const { fullWeightBytes, activeWeightBytes } = draftWeights(cfg);
  const kvBytesPerRequest =
    cfg.draftMode === 'preset' && cfg.draftModel ? kvBytesForContext(cfg.draftModel, contextTokens, kvQuant) : 0;
  const users = Math.max(0, Math.floor(concurrentUsers));
  const kvBytesAllUsers = kvBytesPerRequest * users;
  return {
    weightBytes: fullWeightBytes,
    activeWeightBytes,
    kvBytesPerRequest,
    kvBytesAllUsers,
    totalBytes: fullWeightBytes + kvBytesAllUsers,
  };
}

/**
 * Speculative decode speed: expectedTokensPerStep / (target's bandwidth-bound step + k draft
 * steps). Draft steps reuse the target's own bandwidth/efficiency model (the same GPUs serve
 * both models) rather than a separate tensor-parallel check for the draft's own head count.
 * n-gram/prompt-lookup drafting ('none') has no model forward pass, so its step time is 0.
 */
export function speculativeThroughput(
  cfg: SpeculativeConfig,
  baseline: { perUserTokS: number; efficiency: number },
  draft: { activeWeightBytes: number; kvBytesPerRequest: number },
  concurrentUsers: number,
  hardware: { bandwidthGBs: number; gpuCount: number },
): SpeculativeThroughput {
  const users = Math.max(0, Math.floor(concurrentUsers));
  const targetStepSeconds = baseline.perUserTokS > 0 ? 1 / baseline.perUserTokS : Number.POSITIVE_INFINITY;
  if (!cfg.enabled) {
    return {
      expectedTokensPerStep: 1,
      targetStepSeconds,
      draftStepSeconds: 0,
      verifyStepSeconds: targetStepSeconds,
      perUserTokS: baseline.perUserTokS,
      aggregateTokS: baseline.perUserTokS * users,
      multiplier: 1,
    };
  }
  const k = Math.max(0, Math.floor(cfg.k));
  const e = expectedTokensPerStep(cfg.alpha, k);
  const draftStepSeconds =
    cfg.draftMode === 'none'
      ? 0
      : (() => {
          const draftDecode = decodeThroughput({
            activeWeightBytes: draft.activeWeightBytes,
            kvBytesPerRequest: draft.kvBytesPerRequest,
            concurrentUsers: users,
            bandwidthGBs: hardware.bandwidthGBs,
            gpuCount: hardware.gpuCount,
            efficiency: baseline.efficiency,
          });
          return draftDecode.perUserTokS > 0 ? 1 / draftDecode.perUserTokS : Number.POSITIVE_INFINITY;
        })();
  const verifyStepSeconds = targetStepSeconds + k * draftStepSeconds;
  const perUserTokS = Number.isFinite(verifyStepSeconds) && verifyStepSeconds > 0 ? e / verifyStepSeconds : 0;
  const multiplier = baseline.perUserTokS > 0 ? perUserTokS / baseline.perUserTokS : 0;
  return {
    expectedTokensPerStep: e,
    targetStepSeconds,
    draftStepSeconds,
    verifyStepSeconds,
    perUserTokS,
    aggregateTokS: perUserTokS * users,
    multiplier,
  };
}
