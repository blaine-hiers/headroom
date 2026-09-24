import { describe, expect, it } from 'vitest';
import { MAX_CATALOG_CONTEXT, MAX_GPUS, MAX_USERS, MIN_CONTEXT } from './limits';
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
