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
    const model = makeSpec(); // no ffn
    const r = checkTensorParallelSplit(model, 3);
    expect(r.checkable).toBe(false);
    expect(r.headsDivisible).toBe(true);
    expect(r.suggestedGpuCounts).toEqual([]);
  });

  it('replicates KV heads when numKvHeads < gpuCount', () => {
    const model = makeSpec({ numKvHeads: 2 });
    const r = checkTensorParallelSplit(model, 8);
    expect(r.kvHeadsReplicated).toBe(true);
    expect(r.effectiveKvHeads).toBe(8);
    expect(r.kvReplicationFactor).toBe(4);
  });

  it('does not replicate KV heads when numKvHeads >= gpuCount', () => {
    const model = makeSpec({ numKvHeads: 8 });
    const r = checkTensorParallelSplit(model, 8);
    expect(r.kvHeadsReplicated).toBe(false);
    expect(r.effectiveKvHeads).toBe(8);
    expect(r.kvReplicationFactor).toBe(1);
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
