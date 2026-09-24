import { describe, expect, it, beforeEach } from 'vitest';
import type { ModelSpec } from '../lib';
import { addRecent, clearRecents, getRecents, removeRecent } from './recents';

const mockModel = (id: string, name: string = id): ModelSpec => ({
  id,
  name,
  params: 70e9,
  activeParams: 70e9,
  numLayers: 80,
  attention: 'mha_gqa',
  numKvHeads: 8,
  headDim: 128,
  maxPositionEmbeddings: 8192,
  hiddenSize: 8192,
  vocabSize: 128256,
  nativeDtype: 'bf16',
  source: 'hf',
  warnings: [],
});

beforeEach(() => {
  localStorage.clear();
});

describe('recents list logic', () => {
  it('getRecents returns empty array when nothing is stored', () => {
    expect(getRecents()).toEqual([]);
  });

  it('addRecent stores a model and getRecents returns it', () => {
    const model = mockModel('meta-llama/Llama-3.3-70B-Instruct');
    addRecent(model);
    expect(getRecents()).toEqual([model]);
  });

  it('addRecent puts the newest model first', () => {
    const model1 = mockModel('model1');
    const model2 = mockModel('model2');
    addRecent(model1);
    addRecent(model2);
    expect(getRecents()).toEqual([model2, model1]);
  });

  it('addRecent deduplicates by id, moving the model to the front', () => {
    const model1 = mockModel('model1');
    const model2 = mockModel('model2');
    addRecent(model1);
    addRecent(model2);
    addRecent(model1); // add model1 again
    expect(getRecents()).toEqual([model1, model2]);
  });

  it('addRecent caps the list at 8 items', () => {
    for (let i = 0; i < 10; i++) {
      addRecent(mockModel(`model${i}`));
    }
    const recents = getRecents();
    expect(recents).toHaveLength(8);
    // The last 2 should be dropped (oldest)
    expect(recents.map((m) => m.id)).toEqual([
      'model9',
      'model8',
      'model7',
      'model6',
      'model5',
      'model4',
      'model3',
      'model2',
    ]);
  });

  it('removeRecent removes a specific model by id', () => {
    const model1 = mockModel('model1');
    const model2 = mockModel('model2');
    const model3 = mockModel('model3');
    addRecent(model1);
    addRecent(model2);
    addRecent(model3);
    removeRecent('model2');
    expect(getRecents()).toEqual([model3, model1]);
  });

  it('removeRecent clears localStorage when removing the last item', () => {
    const model = mockModel('model1');
    addRecent(model);
    removeRecent('model1');
    expect(localStorage.getItem('headroom.recents')).toBe(null);
    expect(getRecents()).toEqual([]);
  });

  it('clearRecents removes all recents', () => {
    addRecent(mockModel('model1'));
    addRecent(mockModel('model2'));
    addRecent(mockModel('model3'));
    clearRecents();
    expect(getRecents()).toEqual([]);
    expect(localStorage.getItem('headroom.recents')).toBe(null);
  });

  it('handles corrupted localStorage gracefully', () => {
    localStorage.setItem('headroom.recents', 'not valid json');
    expect(getRecents()).toEqual([]);
  });

  it('handles missing fields in stored models gracefully', () => {
    const incomplete = { id: 'test', name: 'test' }; // missing required fields
    localStorage.setItem('headroom.recents', JSON.stringify([incomplete]));
    expect(getRecents()).toEqual([]);
  });

  it('tolerates localStorage unavailable (throws)', () => {
    const originalStorage = window.localStorage;
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: () => {
          throw new Error('Storage unavailable');
        },
        setItem: () => {
          throw new Error('Storage unavailable');
        },
        removeItem: () => {
          throw new Error('Storage unavailable');
        },
      },
      configurable: true,
    });
    // Should not throw
    expect(() => getRecents()).not.toThrow();
    expect(() => addRecent(mockModel('test'))).not.toThrow();
    expect(() => removeRecent('test')).not.toThrow();
    expect(() => clearRecents()).not.toThrow();
    // Restore original localStorage
    Object.defineProperty(window, 'localStorage', {
      value: originalStorage,
      configurable: true,
    });
  });
});
