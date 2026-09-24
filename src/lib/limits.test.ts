import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { clampWorkloadFor, MAX_CATALOG_CONTEXT, MAX_GPUS, MAX_USERS, maxContextFor, MIN_CONTEXT } from './limits';
import { MODEL_CATALOG } from './presets/catalog';

describe('limits', () => {
  it('MAX_CATALOG_CONTEXT is the largest maxPositionEmbeddings among the bundled catalog entries', () => {
    const expected = Math.max(...MODEL_CATALOG.map((e) => e.spec.maxPositionEmbeddings));
    expect(MAX_CATALOG_CONTEXT).toBe(expected);
    expect(MAX_CATALOG_CONTEXT).toBeGreaterThanOrEqual(MIN_CONTEXT);
  });

  it('re-exports the same bounds the Calculator UI uses', () => {
    expect(MIN_CONTEXT).toBe(256);
    expect(MAX_USERS).toBe(512);
    expect(MAX_GPUS).toBe(16);
  });
});

describe('maxContextFor', () => {
  it('matches the model maxPositionEmbeddings, never below MIN_CONTEXT', () => {
    expect(maxContextFor(makeSpec({ maxPositionEmbeddings: 131072 }))).toBe(131072);
    expect(maxContextFor(makeSpec({ maxPositionEmbeddings: 0 }))).toBe(MIN_CONTEXT);
  });
});

// Regression for #27 review item 1: HardwareSizing rows used to run calculate() at the raw,
// UNclamped workload while "Use" (loadPartial) clamps to the model's own max context and to
// MAX_USERS, so a row's numbers could disagree with what "Use" actually loads. clampWorkloadFor
// is the one place both sides now clamp through.
describe('clampWorkloadFor', () => {
  it('clamps context down to the model max and a negative user count up to 1', () => {
    const model = makeSpec({ maxPositionEmbeddings: 4096 });
    expect(clampWorkloadFor(model, { contextTokens: 131072, concurrentUsers: -3 })).toEqual({ contextTokens: 4096, concurrentUsers: 1 });
  });

  it('clamps a too-small context up to MIN_CONTEXT and an oversized user count down to MAX_USERS', () => {
    const model = makeSpec({ maxPositionEmbeddings: 131072 });
    expect(clampWorkloadFor(model, { contextTokens: 0, concurrentUsers: 999_999_999 })).toEqual({ contextTokens: MIN_CONTEXT, concurrentUsers: MAX_USERS });
  });

  it('falls back to the minimum for a non-finite value', () => {
    const model = makeSpec({ maxPositionEmbeddings: 131072 });
    expect(clampWorkloadFor(model, { contextTokens: NaN, concurrentUsers: Infinity })).toEqual({ contextTokens: MIN_CONTEXT, concurrentUsers: 1 });
  });

  it('is a no-op for a workload already in range', () => {
    const model = makeSpec({ maxPositionEmbeddings: 131072 });
    expect(clampWorkloadFor(model, { contextTokens: 8192, concurrentUsers: 32 })).toEqual({ contextTokens: 8192, concurrentUsers: 32 });
  });
});
