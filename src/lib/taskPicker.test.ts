import { describe, expect, it } from 'vitest';
import { makeSpec } from './__fixtures__/makeSpec';
import { DEFAULT_TASK_PICKER_CONSTRAINTS, rankModelsForTask } from './taskPicker';
import type { TaskPickerConstraints } from './taskPicker';
import type { CatalogEntry, CatalogMeta } from './presets/catalog';
import type { ModelSpec } from './types';

function entry(specOverrides: Partial<ModelSpec>, metaOverrides: Partial<CatalogMeta>): CatalogEntry {
  const spec = makeSpec({ id: metaOverrides.family ?? 'test/model', name: metaOverrides.family ?? 'Test Model', ...specOverrides });
  const meta: CatalogMeta = { provider: 'test', family: 'Test', tags: ['chat'], license: 'apache-2.0', releaseDate: '2025-01', ...metaOverrides };
  return { spec, meta };
}

function constraints(overrides: Partial<TaskPickerConstraints> = {}): TaskPickerConstraints {
  return { ...DEFAULT_TASK_PICKER_CONSTRAINTS, gpuName: 'RTX 4090', gpuCount: 1, ...overrides };
}

describe('rankModelsForTask', () => {
  it('filters out models not tagged with the requested task', () => {
    const catalog = [
      entry({ id: 'a', name: 'A' }, { tags: ['coding'] }),
      entry({ id: 'b', name: 'B' }, { tags: ['chat'] }),
    ];
    const { rows } = rankModelsForTask(catalog, constraints({ task: 'chat' }));
    expect(rows.map((r) => r.entry.spec.id)).toEqual(['b']);
  });

  it('filters out models whose maxPositionEmbeddings is below the needed context', () => {
    const catalog = [
      entry({ id: 'short', name: 'Short', maxPositionEmbeddings: 4096 }, {}),
      entry({ id: 'long', name: 'Long', maxPositionEmbeddings: 131072 }, {}),
    ];
    const { rows } = rankModelsForTask(catalog, constraints({ contextTokens: 32768 }));
    expect(rows.map((r) => r.entry.spec.id)).toEqual(['long']);
  });

  it('permissive license filter keeps only the explicit permissive list', () => {
    const catalog = [
      entry({ id: 'mit', name: 'MIT model' }, { license: 'mit' }),
      entry({ id: 'apache', name: 'Apache model' }, { license: 'apache-2.0' }),
      entry({ id: 'llama', name: 'Llama model' }, { license: 'llama3.1' }),
    ];
    const { rows } = rankModelsForTask(catalog, constraints({ licenseFilter: 'permissive' }));
    expect(new Set(rows.map((r) => r.entry.spec.id))).toEqual(new Set(['mit', 'apache']));
  });

  it('excludes MoE models when MoE is not allowed', () => {
    const catalog = [
      entry({ id: 'dense', name: 'Dense', params: 8e9, activeParams: 8e9 }, {}),
      entry({ id: 'moe', name: 'MoE', params: 30e9, activeParams: 3e9, moe: { numExperts: 8, expertsPerToken: 2, sharedExperts: 0 } }, {}),
    ];
    const { rows } = rankModelsForTask(catalog, constraints({ allowMoe: false }));
    expect(rows.map((r) => r.entry.spec.id)).toEqual(['dense']);
  });

  it('returns an empty result with a hint when nothing matches the filters', () => {
    const catalog = [entry({ id: 'a', name: 'A' }, { tags: ['coding'] })];
    const result = rankModelsForTask(catalog, constraints({ task: 'vision' }));
    expect(result.rows).toEqual([]);
    expect(result.hint).toMatch(/vision/);
  });

  it('ranks a model that runs above one that does not, even when the non-running one is bigger', () => {
    const catalog = [
      // 2 trillion params at BF16 does not fit any single RTX 4090.
      entry({ id: 'huge', name: 'Huge', params: 2e12, activeParams: 2e12 }, {}),
      entry({ id: 'small', name: 'Small', params: 8e9, activeParams: 8e9 }, {}),
    ];
    const { rows } = rankModelsForTask(catalog, constraints());
    expect(rows[0].entry.spec.id).toBe('small');
    expect(rows[0].result.runs).toBe(true);
    expect(rows[1].entry.spec.id).toBe('huge');
    expect(rows[1].result.runs).toBe(false);
  });

  it('among models that run, prefers larger total params first', () => {
    const catalog = [
      entry({ id: 'small', name: 'Small', params: 3e9, activeParams: 3e9 }, {}),
      entry({ id: 'bigger', name: 'Bigger', params: 8e9, activeParams: 8e9 }, {}),
    ];
    const { rows } = rankModelsForTask(catalog, constraints());
    expect(rows.map((r) => r.entry.spec.id)).toEqual(['bigger', 'small']);
  });

  it('ties on params break by active params, then release date (newer first)', () => {
    const catalog = [
      entry({ id: 'older', name: 'Older', params: 8e9, activeParams: 8e9 }, { releaseDate: '2024-01' }),
      entry({ id: 'newer', name: 'Newer', params: 8e9, activeParams: 8e9 }, { releaseDate: '2025-06' }),
    ];
    const { rows } = rankModelsForTask(catalog, constraints());
    expect(rows.map((r) => r.entry.spec.id)).toEqual(['newer', 'older']);
  });

  it('limits the result to the top 8 rows', () => {
    const catalog = Array.from({ length: 12 }, (_, i) => entry({ id: `m${i}`, name: `M${i}`, params: (i + 1) * 1e9, activeParams: (i + 1) * 1e9 }, {}));
    const { rows } = rankModelsForTask(catalog, constraints());
    expect(rows).toHaveLength(8);
  });

  it('a named GPU preset is used as-is for every row', () => {
    const catalog = [entry({ id: 'a', name: 'A', params: 8e9, activeParams: 8e9 }, {})];
    const { rows } = rankModelsForTask(catalog, constraints({ gpuName: 'H100 SXM', gpuCount: 1 }));
    expect(rows[0].hardware.gpuName).toBe('H100 SXM');
  });

  it('"any" hardware finds a fitting preset for a small model', () => {
    const catalog = [entry({ id: 'a', name: 'A', params: 1e9, activeParams: 1e9 }, {})];
    const { rows } = rankModelsForTask(catalog, constraints({ gpuName: 'any' }));
    expect(rows[0].result.runs).toBe(true);
    expect(rows[0].hardware.gpuName).not.toBe('Custom');
  });

  it('builds a one-line reason mentioning headroom, tok/s, license and context', () => {
    const catalog = [entry({ id: 'a', name: 'A', params: 8e9, activeParams: 8e9, maxPositionEmbeddings: 131072 }, { license: 'apache-2.0' })];
    const { rows } = rankModelsForTask(catalog, constraints());
    expect(rows[0].reason).toMatch(/tok\/s/);
    expect(rows[0].reason).toMatch(/Apache-2\.0/);
    expect(rows[0].reason).toMatch(/128K ctx/);
  });
});
