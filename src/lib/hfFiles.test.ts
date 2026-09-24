import { describe, expect, it, vi } from 'vitest';
import { fromBase64 } from './__fixtures__/base64';
import qwenMoeB64 from './__fixtures__/qwen3-30b-a3b-q4_k_m.gguf.b64?raw';
import qwenApiRaw from './__fixtures__/qwen2.5-7b-instruct.api.json?raw';
import qwenRaw from './__fixtures__/qwen2.5-7b-instruct.json?raw';
import { HF_ERRORS, preQuantOf } from './hf';
import {
  defaultGgufOption,
  fetchRepo,
  GGUF_HEADER_BYTES,
  ggufOptions,
  ggufWeightQuant,
  noQuantMatchNote,
  parseRepoListing,
  preQuantWeightQuant,
  quantFromFilename,
  safetensorsBytes,
} from './hfFiles';
import type { RepoFile } from './hfFiles';

const j = (raw: string): unknown => JSON.parse(raw);

function res(status: number, body: unknown): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status });
}

describe('quantFromFilename', () => {
  it('reads the llama.cpp quant tag, ignoring shard suffixes and folders', () => {
    expect(quantFromFilename('Qwen2.5-7B-Instruct-Q4_K_M.gguf')).toBe('Q4_K_M');
    expect(quantFromFilename('Qwen_Qwen3-30B-A3B-IQ4_XS.gguf')).toBe('IQ4_XS');
    expect(quantFromFilename('Q2_K/Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf')).toBe('Q2_K');
    expect(quantFromFilename('openai_gpt-oss-20b-MXFP4.gguf')).toBe('MXFP4');
    expect(quantFromFilename('model-bf16.gguf')).toBe('BF16');
    expect(quantFromFilename('UD-Q2_K_XL/DeepSeek-V3.1-UD-Q2_K_XL-00001-of-00006.gguf')).toBe('Q2_K_XL');
    expect(quantFromFilename('model.gguf')).toBeUndefined();
  });
});

describe('ggufOptions', () => {
  const files: RepoFile[] = [
    { path: '.gitattributes', size: 3000 },
    { path: 'm-Q8_0.gguf', size: 8e9 },
    { path: 'm-Q4_K_M.gguf', size: 4.6e9 },
    { path: 'mmproj-m-f16.gguf', size: 6e8 },
    { path: 'imatrix.gguf', size: 1e6 },
    { path: 'BF16/m-BF16-00002-of-00002.gguf', size: 5e9 },
    { path: 'BF16/m-BF16-00001-of-00002.gguf', size: 10e9 },
  ];

  it('groups split shards, sums their sizes, and drops projector and imatrix files', () => {
    expect(ggufOptions(files)).toEqual([
      { path: 'm-Q4_K_M.gguf', label: 'Q4_K_M', bytes: 4.6e9, shards: 1 },
      { path: 'm-Q8_0.gguf', label: 'Q8_0', bytes: 8e9, shards: 1 },
      { path: 'BF16/m-BF16-00001-of-00002.gguf', label: 'BF16', bytes: 15e9, shards: 2 },
    ]);
  });

  it('defaults to Q4_K_M, else the smallest file', () => {
    expect(defaultGgufOption(ggufOptions(files))?.label).toBe('Q4_K_M');
    expect(defaultGgufOption(ggufOptions(files.filter((f) => !f.path.includes('Q4'))))?.label).toBe('Q8_0');
    expect(defaultGgufOption([])).toBeUndefined();
  });
});

describe('repo listing helpers', () => {
  it('parseRepoListing reads sibling sizes (or lfs sizes) and gguf.total', () => {
    const l = parseRepoListing({
      siblings: [{ rfilename: 'a.gguf', size: 10 }, { rfilename: 'b.gguf', lfs: { size: 20 } }, { rfilename: 'c.txt' }],
      gguf: { total: 7_615_616_512, architecture: 'qwen2' },
    });
    expect(l).toEqual({ files: [{ path: 'a.gguf', size: 10 }, { path: 'b.gguf', size: 20 }], ggufParams: 7_615_616_512 });
    expect(parseRepoListing(null)).toEqual({ files: [] });
  });

  it('safetensorsBytes sums top-level weight files only', () => {
    expect(
      safetensorsBytes([
        { path: 'model-00001-of-00002.safetensors', size: 3996422976 },
        { path: 'model-00002-of-00002.safetensors', size: 1574406784 },
        { path: 'consolidated.safetensors', size: 5e9 },
        { path: 'original/model.safetensors', size: 5e9 },
        { path: 'config.json', size: 841 },
      ]),
    ).toBe(3996422976 + 1574406784);
    expect(safetensorsBytes([{ path: 'x.gguf', size: 1 }])).toBeUndefined();
  });

  it('ggufWeightQuant: table tags exactly, others by effective bits, else by tag family', () => {
    expect(ggufWeightQuant('Q4_K_M', 1, 1)).toEqual({ quant: 'q4_k_m', exact: true });
    expect(ggufWeightQuant('IQ2_M', 2.78e9, 7.6e9)).toEqual({ quant: 'q2_k', exact: false }); // 2.93 bits/weight
    // No params: the tag family decides.
    expect(ggufWeightQuant('IQ2_M', 2.78e9, 0)).toEqual({ quant: 'q2_k', exact: false });
    expect(ggufWeightQuant('Q2_K_L', 1, 0)).toEqual({ quant: 'q2_k', exact: false });
    expect(ggufWeightQuant('Q4_K_XL', 1, 0)).toEqual({ quant: 'q4_k_m', exact: false });
    expect(ggufWeightQuant('Q5_K_S', 1, 0)).toEqual({ quant: 'q5_k_m', exact: false });
    expect(ggufWeightQuant('MXFP4', 1, 0)).toEqual({ quant: 'q4_0', exact: false });
    expect(ggufWeightQuant('model', 1, 0)).toBeUndefined();
  });

  it('preQuantWeightQuant: bitsandbytes 4-bit is NF4 and 8-bit is INT8, whatever the bytes', () => {
    expect(preQuantWeightQuant({ method: 'bitsandbytes', bits: 4 }, 5.5e9, 7.6e9)).toEqual({ quant: 'nf4', exact: true });
    expect(preQuantWeightQuant({ method: 'bitsandbytes', bits: 8 }, 5.5e9, 7.6e9)).toEqual({ quant: 'q8_0', exact: true });
    expect(preQuantWeightQuant({ method: 'awq', bits: 4 }, 1, 1)).toEqual({ quant: 'awq_gptq_4bit', exact: true });
    expect(preQuantWeightQuant({ method: 'fp8' }, 1, 1)).toEqual({ quant: 'fp8', exact: true });
    expect(preQuantWeightQuant({ method: 'mxfp4' }, 1, 0)).toBeUndefined();
  });

  it('preQuantOf reads bitsandbytes load_in_4bit / load_in_8bit', () => {
    const bnb = (q: object) => preQuantOf({ quantization_config: { quant_method: 'bitsandbytes', ...q } });
    expect(bnb({ load_in_4bit: true, bnb_4bit_quant_type: 'nf4' })).toEqual({ method: 'bitsandbytes', bits: 4 });
    expect(bnb({ load_in_8bit: true, load_in_4bit: false, bnb_4bit_quant_type: 'fp4' })).toEqual({ method: 'bitsandbytes', bits: 8 });
    expect(bnb({ bnb_4bit_quant_type: 'nf4' })).toEqual({ method: 'bitsandbytes', bits: 4 });
  });
});

const GGUF_REPO = 'bartowski/Qwen_Qwen3-30B-A3B-GGUF';
const ggufListing = {
  siblings: [
    { rfilename: 'Qwen_Qwen3-30B-A3B-Q4_K_M.gguf', size: 18_556_686_080 },
    { rfilename: 'Qwen_Qwen3-30B-A3B-Q8_0.gguf', size: 32_483_935_968 },
    { rfilename: 'README.md', size: 1000 },
  ],
  gguf: { total: 30_532_122_624, architecture: 'qwen3moe', context_length: 32768 },
};

function ggufFetch(header: () => Response) {
  return vi.fn(async (url: string | URL | Request) => {
    const u = String(url);
    if (u.includes('?blobs=true')) return res(200, ggufListing);
    if (u.includes('/api/models/')) return res(200, { id: GGUF_REPO });
    if (u.endsWith('/config.json')) return res(404, 'Entry not found');
    return header();
  });
}

describe('fetchRepo', () => {
  it('GGUF repo: lists the files, reads the Q4_K_M header with a Range request, uses its exact size', async () => {
    const fetchImpl = ggufFetch(() => new Response(fromBase64(qwenMoeB64), { status: 206 }));
    const r = await fetchRepo(GGUF_REPO, undefined, undefined, fetchImpl as typeof fetch);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.spec).toMatchObject({ id: GGUF_REPO, numLayers: 48, params: 30_532_122_624, source: 'hf' });
    expect(r.spec.fileWeights).toEqual({ bytes: 18_556_686_080, label: 'Q4_K_M GGUF', quant: 'q4_k_m' });
    expect(r.gguf?.selected).toBe('Qwen_Qwen3-30B-A3B-Q4_K_M.gguf');
    expect(r.gguf?.options.map((o) => o.label)).toEqual(['Q4_K_M', 'Q8_0']);
    const headerCall = fetchImpl.mock.calls.find((c) => String(c[0]).endsWith('.gguf')) as unknown as [string, RequestInit];
    expect(headerCall[0]).toBe(`https://huggingface.co/${GGUF_REPO}/resolve/main/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf`);
    expect((headerCall[1].headers as Record<string, string>).Range).toBe(`bytes=0-${GGUF_HEADER_BYTES - 1}`);
    expect(fetchImpl.mock.calls.some((c) => String(c[0]) === `https://huggingface.co/api/models/${GGUF_REPO}?blobs=true&expand[]=siblings&expand[]=gguf`)).toBe(true);
  });

  it('GGUF repo: a chosen file is read instead of the default', async () => {
    const fetchImpl = ggufFetch(() => new Response(fromBase64(qwenMoeB64), { status: 206 }));
    const r = await fetchRepo(GGUF_REPO, undefined, 'Qwen_Qwen3-30B-A3B-Q8_0.gguf', fetchImpl as typeof fetch);
    expect(r.ok && r.spec.fileWeights).toEqual({ bytes: 32_483_935_968, label: 'Q8_0 GGUF', quant: 'q8_0' });
  });

  it('GGUF repo: a server that ignores Range is read only up to the header size', async () => {
    const buf = new Uint8Array(GGUF_HEADER_BYTES + 5000);
    buf.set(new Uint8Array(fromBase64(qwenMoeB64)));
    const r = await fetchRepo(GGUF_REPO, undefined, undefined, ggufFetch(() => new Response(buf, { status: 200 })) as typeof fetch);
    expect(r.ok && r.spec.numLayers).toBe(48);
  });

  it('GGUF repo: a failed or unreadable header is an error result, never a throw', async () => {
    const cors = await fetchRepo(GGUF_REPO, undefined, undefined, ggufFetch(() => {
      throw new TypeError('Failed to fetch');
    }) as typeof fetch);
    expect(cors.ok).toBe(false);
    if (!cors.ok) expect(cors.error).toMatch(/GGUF.*network or CORS error/);
    const garbage = await fetchRepo(GGUF_REPO, undefined, undefined, ggufFetch(() => res(206, 'not a gguf at all, just text')) as typeof fetch);
    expect(garbage.ok).toBe(false);
    if (!garbage.ok) expect(garbage.error).toMatch(/not a GGUF file/);
    const denied = await fetchRepo(GGUF_REPO, undefined, undefined, ggufFetch(() => res(500, 'no')) as typeof fetch);
    expect(denied.ok).toBe(false);
    if (!denied.ok) expect(denied.error).toMatch(/HTTP 500/);
  });

  it('GGUF repo: a tag outside the table without gguf.total is tied to its closest quant and labelled so', async () => {
    const listing = { siblings: [{ rfilename: 'm-IQ2_M.gguf', size: 11_000_000_000 }] };
    const fetchImpl = async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes('?blobs=true')) return res(200, listing);
      if (u.includes('/api/models/')) return res(200, {});
      if (u.endsWith('/config.json')) return res(404, 'no');
      return new Response(fromBase64(qwenMoeB64), { status: 206 });
    };
    const r = await fetchRepo(GGUF_REPO, undefined, undefined, fetchImpl as typeof fetch);
    expect(r.ok && r.spec.params).toBe(0);
    expect(r.ok && r.spec.fileWeights).toEqual({ bytes: 11e9, label: 'IQ2_M GGUF (closest table quant: Q2_K)', quant: 'q2_k' });
  });

  it('GGUF repo: a file with no recognisable quant tag keeps no file weights and says why', async () => {
    const listing = { siblings: [{ rfilename: 'model.gguf', size: 11_000_000_000 }] };
    const fetchImpl = async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes('?blobs=true')) return res(200, listing);
      if (u.includes('/api/models/')) return res(200, {});
      if (u.endsWith('/config.json')) return res(404, 'no');
      return new Response(fromBase64(qwenMoeB64), { status: 206 });
    };
    const r = await fetchRepo(GGUF_REPO, undefined, undefined, fetchImpl as typeof fetch);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.spec.fileWeights).toBeUndefined();
    expect(r.spec.warnings).toContain(noQuantMatchNote('model.gguf GGUF'));
  });

  it('gated GGUF repo: a 401/403 header read shows the gated hint', async () => {
    const r = await fetchRepo(GGUF_REPO, undefined, undefined, ggufFetch(() => res(401, 'no')) as typeof fetch);
    expect(r).toEqual({ ok: false, error: HF_ERRORS.gated });
  });

  it('GGUF files plus a usable config.json: a failed header read falls back to config.json with a note', async () => {
    const fetchImpl = async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes('?blobs=true')) return res(200, ggufListing);
      if (u.includes('/api/models/')) return res(200, j(qwenApiRaw));
      if (u.endsWith('/config.json')) return res(200, j(qwenRaw));
      throw new TypeError('Failed to fetch');
    };
    const r = await fetchRepo(GGUF_REPO, undefined, undefined, fetchImpl as typeof fetch);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.spec.numLayers).toBe(28);
    expect(r.spec.fileWeights).toBeUndefined();
    expect(r.note).toMatch(/GGUF header not read \(network or CORS error\)/);
  });

  it('pre-quantized safetensors repo: weight bytes from the files, labelled by quant method', async () => {
    const awqConfig = { ...(j(qwenRaw) as object), quantization_config: { quant_method: 'awq', bits: 4, group_size: 128 } };
    const fetchImpl = vi.fn(async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes('?blobs=true'))
        return res(200, {
          siblings: [
            { rfilename: 'config.json', size: 841 },
            { rfilename: 'model-00001-of-00002.safetensors', size: 3996422976 },
            { rfilename: 'model-00002-of-00002.safetensors', size: 1574406784 },
          ],
        });
      if (u.includes('/api/models/')) return res(200, j(qwenApiRaw));
      return res(200, awqConfig);
    });
    const r = await fetchRepo('Qwen/Qwen2.5-7B-Instruct-AWQ', 'hf_abc', undefined, fetchImpl as typeof fetch);
    expect(r.ok && r.spec.fileWeights).toEqual({ bytes: 3996422976 + 1574406784, label: 'AWQ safetensors', quant: 'awq_gptq_4bit' });
    for (const c of fetchImpl.mock.calls as unknown as Array<[string, RequestInit]>) {
      expect((c[1].headers as Record<string, string>).Authorization).toBe('Bearer hf_abc');
    }
  });

  it('plain BF16 repo: no file weights, same spec as fetchModel', async () => {
    const fetchImpl = async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes('?blobs=true')) return res(200, { siblings: [{ rfilename: 'model.safetensors', size: 15e9 }] });
      return u.includes('/api/models/') ? res(200, j(qwenApiRaw)) : res(200, j(qwenRaw));
    };
    const r = await fetchRepo('Qwen/Qwen2.5-7B-Instruct', undefined, undefined, fetchImpl as typeof fetch);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.spec.fileWeights).toBeUndefined();
      expect(r.gguf).toBeUndefined();
    }
  });

  it('failed listing: falls back to fetchModel alone', async () => {
    const fetchImpl = async (url: string | URL | Request) => {
      const u = String(url);
      if (u.includes('?blobs=true')) throw new TypeError('Failed to fetch');
      return u.includes('/api/models/') ? res(200, j(qwenApiRaw)) : res(200, j(qwenRaw));
    };
    const r = await fetchRepo('Qwen/Qwen2.5-7B-Instruct', undefined, undefined, fetchImpl as typeof fetch);
    expect(r.ok && r.spec.params).toBe(7_615_616_512);
    const missing = await fetchRepo('a/b', undefined, undefined, (async () => res(404, 'no')) as typeof fetch);
    expect(missing).toEqual({ ok: false, error: HF_ERRORS.notFound });
  });

  it('empty id → error without fetching', async () => {
    const fetchImpl = vi.fn();
    expect(await fetchRepo(' ', undefined, undefined, fetchImpl as unknown as typeof fetch)).toEqual({ ok: false, error: HF_ERRORS.emptyId });
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
