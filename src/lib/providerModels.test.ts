import { describe, expect, it } from 'vitest';
import { weightBytes } from './weights';
import { catalogByProvider } from './presets/catalog';
import { providerModelRows } from './providerModels';

describe('providerModelRows', () => {
  it('groups one provider\'s catalog entries, in catalog order', () => {
    const rows = providerModelRows('qwen', 'q4_k_m', '');
    expect(rows.map((r) => r.entry.spec.id)).toEqual(catalogByProvider('qwen').map((e) => e.spec.id));
    for (const row of rows) expect(row.entry.meta.provider).toBe('qwen');
  });

  it('computes each row\'s weight size at the given quant from the existing weight maths', () => {
    const quant = 'bf16';
    const rows = providerModelRows('mistral', quant, '');
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row.weightBytes).toBe(weightBytes(row.entry.spec.params, quant));
    }
  });

  it('recomputes the size for a different quant', () => {
    const [row4] = providerModelRows('meta', 'q4_k_m', '');
    const [rowBf16] = providerModelRows('meta', 'bf16', '');
    expect(row4.weightBytes).toBeLessThan(rowBf16.weightBytes);
  });

  it('marks exactly the currently loaded model, if any', () => {
    const [first] = catalogByProvider('meta');
    const rows = providerModelRows('meta', 'q4_k_m', first.spec.id);
    expect(rows.filter((r) => r.isLoaded)).toEqual([expect.objectContaining({ entry: first })]);

    const noneLoaded = providerModelRows('meta', 'q4_k_m', 'not-a-real-id');
    expect(noneLoaded.some((r) => r.isLoaded)).toBe(false);
  });

  it('returns an empty array for an unknown provider', () => {
    expect(providerModelRows('not-a-provider', 'bf16', '')).toEqual([]);
  });
});
