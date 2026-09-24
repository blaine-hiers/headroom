import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { checkTensorParallelSplit, tensorParallelEfficiency } from './tensorParallel';

describe('checkTensorParallelSplit', () => {
  it('is unaffected at gpuCount 1, regardless of head count', () => {
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 65, tieEmbeddings: false } });
    const r = checkTensorParallelSplit(model, 1);
    expect(r.headsDivisible).toBe(true);
    expect(r.suggestedGpuCounts).toEqual([]);
    expect(r.kvHeadsReplicated).toBe(false);
    expect(r.kvReplicationFactor).toBe(1);
  });

  it('flags a head count that cannot split evenly and suggests the nearest valid counts', () => {
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 64, tieEmbeddings: false } });
    const r = checkTensorParallelSplit(model, 3);
    expect(r.checkable).toBe(true);
    expect(r.numAttentionHeads).toBe(64);
    expect(r.headsDivisible).toBe(false);
    expect(r.suggestedGpuCounts).toEqual([1, 2, 4, 8]);
  });

  it('does not flag a head count that splits evenly', () => {
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 64, tieEmbeddings: false } });
    for (const gpuCount of [1, 2, 4, 8]) {
      expect(checkTensorParallelSplit(model, gpuCount).headsDivisible).toBe(true);
    }
  });

  it('is not checkable without an ffn spec, and does not warn falsely', () => {
    const model = makeSpec({ numKvHeads: 1 }); // no ffn; numKvHeads=1 splits/replicates evenly at any gpuCount
    const r = checkTensorParallelSplit(model, 3);
    expect(r.checkable).toBe(false);
    expect(r.headsDivisible).toBe(true);
    expect(r.suggestedGpuCounts).toEqual([]);
  });

  it('replicates KV heads when numKvHeads < gpuCount', () => {
    const model = makeSpec({ numKvHeads: 2 });
    const r = checkTensorParallelSplit(model, 8);
    expect(r.kvHeadsReplicated).toBe(true);
    expect(r.kvHeadsSplitValid).toBe(true); // 8 % 2 === 0: replicates evenly
    expect(r.effectiveKvHeads).toBe(8);
    expect(r.kvReplicationFactor).toBe(4);
  });

  it('does not replicate KV heads when numKvHeads >= gpuCount', () => {
    const model = makeSpec({ numKvHeads: 8 });
    const r = checkTensorParallelSplit(model, 8);
    expect(r.kvHeadsReplicated).toBe(false);
    expect(r.kvHeadsSplitValid).toBe(true);
    expect(r.effectiveKvHeads).toBe(8);
    expect(r.kvReplicationFactor).toBe(1);
  });

  it('flags a KV-head count that can neither split nor replicate evenly, and suggests counts that satisfy both checks', () => {
    // 48 heads (splits evenly across 8 GPUs) but only 3 KV heads: 8 is not a multiple of 3
    // and 3 is not a multiple of 8, so vLLM can't lay the KV heads out evenly either.
    const model = makeSpec({ ffn: { intermediateSize: 1, numAttentionHeads: 48, tieEmbeddings: false }, numKvHeads: 3 });
    const r = checkTensorParallelSplit(model, 8);
    expect(r.headsDivisible).toBe(true);
    expect(r.kvHeadsReplicated).toBe(true);
    expect(r.kvHeadsSplitValid).toBe(false);
    expect(r.kvReplicationFactor).toBeCloseTo(8 / 3);
    expect(r.suggestedGpuCounts).toEqual([1, 3, 6, 12]);
    for (const g of r.suggestedGpuCounts) {
      expect(48 % g).toBe(0);
      expect(g <= 3 ? 3 % g === 0 : g % 3 === 0).toBe(true);
    }
  });

  it('a KV-head count that splits evenly is not flagged, even when replicated', () => {
    const model = makeSpec({ numKvHeads: 3 });
    for (const gpuCount of [1, 3, 6, 9]) {
      expect(checkTensorParallelSplit(model, gpuCount).kvHeadsSplitValid).toBe(true);
    }
  });

  it('MLA models are never flagged or scaled for KV-head replication, even with a small numKvHeads and multiple GPUs', () => {
    const model = makeSpec({ attention: 'mla', numKvHeads: 2, kvLoraRank: 512, qkRopeHeadDim: 64 });
    const r = checkTensorParallelSplit(model, 8);
    expect(r.kvHeadsReplicated).toBe(false);
    expect(r.kvHeadsSplitValid).toBe(true);
    expect(r.effectiveKvHeads).toBe(2);
    expect(r.kvReplicationFactor).toBe(1);
  });

  it('MLA models still get the attention-head divisibility check', () => {
    const model = makeSpec({
      attention: 'mla',
      numKvHeads: 1,
      kvLoraRank: 512,
      qkRopeHeadDim: 64,
      ffn: { intermediateSize: 1, numAttentionHeads: 64, tieEmbeddings: false },
    });
    const r = checkTensorParallelSplit(model, 3);
    expect(r.checkable).toBe(true);
    expect(r.headsDivisible).toBe(false);
    expect(r.suggestedGpuCounts).toEqual([1, 2, 4, 8]);
  });
});

describe('tensorParallelEfficiency', () => {
  it('is 1 at gpuCount 1 (single GPU unaffected)', () => {
    expect(tensorParallelEfficiency(1)).toBe(1);
    expect(tensorParallelEfficiency(0)).toBe(1);
  });

  it('applies ×0.9 per doubling of GPU count', () => {
    expect(tensorParallelEfficiency(2)).toBeCloseTo(0.9);
    expect(tensorParallelEfficiency(4)).toBeCloseTo(0.81);
    expect(tensorParallelEfficiency(8)).toBeCloseTo(0.729);
  });
});
