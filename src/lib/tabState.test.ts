import { describe, expect, it } from 'vitest';
import { decodeTab, setTabParam } from './tabState';

describe('decodeTab', () => {
  it('defaults to calculator with no tab key (every old link)', () => {
    expect(decodeTab('')).toBe('calculator');
    expect(decodeTab('?id=x&n=Foo')).toBe('calculator');
  });

  it('accepts a leading "?" or not', () => {
    expect(decodeTab('tab=planner')).toBe('planner');
    expect(decodeTab('?tab=planner')).toBe('planner');
  });

  it('falls back to calculator on an unrecognized value', () => {
    expect(decodeTab('?tab=nope')).toBe('calculator');
  });

  it('does not confuse a compare column key for the tab', () => {
    expect(decodeTab('?id=x&c2=id%3Dy&tab=planner')).toBe('planner');
  });
});

describe('setTabParam', () => {
  it('omits the key for the default tab, so a Calculator link never grows', () => {
    const params = new URLSearchParams('id=x');
    setTabParam(params, 'calculator');
    expect(params.toString()).toBe('id=x');
  });

  it('removes an existing tab key when switching back to calculator', () => {
    const params = new URLSearchParams('id=x&tab=planner');
    setTabParam(params, 'calculator');
    expect(params.toString()).toBe('id=x');
  });

  it('sets the key for a non-default tab', () => {
    const params = new URLSearchParams('id=x');
    setTabParam(params, 'planner');
    expect(params.toString()).toBe('id=x&tab=planner');
  });
});
