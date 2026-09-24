import { describe, expect, it } from 'vitest';
import { fuzzyMatches, listAuthorModels, searchHub } from './hfSearch';

function res(status: number, body: unknown): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status });
}

describe('searchHub', () => {
  it('returns hits with id, downloads and a gated flag derived from the API field', async () => {
    const fetchImpl = (async (url: string) => {
      expect(url).toContain('https://huggingface.co/api/models?search=llama');
      expect(url).toContain('limit=10');
      expect(url).toContain('sort=downloads');
      expect(url).toContain('expand[]=gated');
      return res(200, [
        { id: 'meta-llama/Llama-3.1-8B-Instruct', downloads: 6225344, gated: 'manual' },
        { id: 'NousResearch/Llama-2-7b-hf', downloads: 12345, gated: false },
      ]);
    }) as unknown as typeof fetch;

    const hits = await searchHub('llama', undefined, fetchImpl);
    expect(hits).toEqual([
      { id: 'meta-llama/Llama-3.1-8B-Instruct', downloads: 6225344, gated: true },
      { id: 'NousResearch/Llama-2-7b-hf', downloads: 12345, gated: false },
    ]);
  });

  it('sends the token as a Bearer header when one is given', async () => {
    let sentHeaders: Record<string, string> | undefined;
    const fetchImpl = (async (_url: string, init?: RequestInit) => {
      sentHeaders = init?.headers as Record<string, string>;
      return res(200, []);
    }) as unknown as typeof fetch;

    await searchHub('llama', 'hf_abc', fetchImpl);
    expect(sentHeaders?.Authorization).toBe('Bearer hf_abc');
  });

  it('omits the Authorization header when no token is stored', async () => {
    let sentHeaders: Record<string, string> | undefined;
    const fetchImpl = (async (_url: string, init?: RequestInit) => {
      sentHeaders = init?.headers as Record<string, string>;
      return res(200, []);
    }) as unknown as typeof fetch;

    await searchHub('llama', undefined, fetchImpl);
    expect(sentHeaders?.Authorization).toBeUndefined();
  });

  it('returns an empty array for an empty query, without calling fetch', async () => {
    let called = false;
    const fetchImpl = (async () => {
      called = true;
      return res(200, []);
    }) as unknown as typeof fetch;

    expect(await searchHub('   ', undefined, fetchImpl)).toEqual([]);
    expect(called).toBe(false);
  });

  it('returns an empty array when the API responds with no matches', async () => {
    const fetchImpl = (async () => res(200, [])) as unknown as typeof fetch;
    expect(await searchHub('zzzznomodellikethis', undefined, fetchImpl)).toEqual([]);
  });

  it('is silent on a non-OK response (rate limit, etc.)', async () => {
    const fetchImpl = (async () => res(429, 'rate limited')) as unknown as typeof fetch;
    expect(await searchHub('llama', undefined, fetchImpl)).toEqual([]);
  });

  it('is silent on a network failure (offline)', async () => {
    const fetchImpl = (async () => {
      throw new Error('network down');
    }) as unknown as typeof fetch;
    expect(await searchHub('llama', undefined, fetchImpl)).toEqual([]);
  });

  it('is silent on unparsable JSON', async () => {
    const fetchImpl = (async () => new Response('not json', { status: 200 })) as unknown as typeof fetch;
    expect(await searchHub('llama', undefined, fetchImpl)).toEqual([]);
  });

  it('is silent when the response body is not an array', async () => {
    const fetchImpl = (async () => res(200, { error: 'nope' })) as unknown as typeof fetch;
    expect(await searchHub('llama', undefined, fetchImpl)).toEqual([]);
  });
});

describe('listAuthorModels', () => {
  it('queries by author, sorted by downloads, with no search term', async () => {
    const fetchImpl = (async (url: string) => {
      expect(url).toContain('https://huggingface.co/api/models?author=Qwen');
      expect(url).toContain('sort=downloads');
      expect(url).toContain('limit=20');
      return res(200, [{ id: 'Qwen/Qwen3-32B', downloads: 500000, gated: false }]);
    }) as unknown as typeof fetch;

    expect(await listAuthorModels('Qwen', undefined, fetchImpl)).toEqual([
      { id: 'Qwen/Qwen3-32B', downloads: 500000, gated: false },
    ]);
  });

  it('returns an empty array for an empty author, without calling fetch', async () => {
    let called = false;
    const fetchImpl = (async () => {
      called = true;
      return res(200, []);
    }) as unknown as typeof fetch;
    expect(await listAuthorModels('  ', undefined, fetchImpl)).toEqual([]);
    expect(called).toBe(false);
  });

  it('is silent on a non-OK response, a network failure, or unparsable JSON', async () => {
    expect(await listAuthorModels('Qwen', undefined, (async () => res(503, 'down')) as unknown as typeof fetch)).toEqual([]);
    expect(
      await listAuthorModels(
        'Qwen',
        undefined,
        (async () => {
          throw new Error('offline');
        }) as unknown as typeof fetch,
      ),
    ).toEqual([]);
    expect(await listAuthorModels('Qwen', undefined, (async () => new Response('not json', { status: 200 })) as unknown as typeof fetch)).toEqual([]);
  });
});

describe('fuzzyMatches', () => {
  it('matches a plain substring of the id or name', () => {
    expect(fuzzyMatches('llama', 'meta-llama/Llama-3.1-8B-Instruct', 'Llama 3.1 8B')).toBe(true);
    expect(fuzzyMatches('3.1 8b', 'meta-llama/Llama-3.1-8B-Instruct', 'Llama 3.1 8B')).toBe(true);
  });

  it('matches an in-order subsequence when there is no direct substring', () => {
    expect(fuzzyMatches('lla31', 'meta-llama/Llama-3.1-8B-Instruct', 'Llama 3.1 8B')).toBe(true);
  });

  it('does not match out-of-order or missing characters', () => {
    expect(fuzzyMatches('zzz', 'meta-llama/Llama-3.1-8B-Instruct', 'Llama 3.1 8B')).toBe(false);
    expect(fuzzyMatches('13lla', 'meta-llama/Llama-3.1-8B-Instruct', 'Llama 3.1 8B')).toBe(false);
  });

  it('returns false for an empty query', () => {
    expect(fuzzyMatches('', 'org/model', 'Model')).toBe(false);
  });
});
