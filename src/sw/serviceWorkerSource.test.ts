import { describe, expect, it } from 'vitest';
import { serviceWorkerSource } from './serviceWorkerSource';

const ORIGIN = 'https://example.test';
const BASE = '/headroom/';
const INDEX_HTML = '<!doctype html><div id="root"></div>';

type Listener = (event: unknown) => void;

/** Runs the generated worker against in-memory caches and a switchable fake network. */
function harness() {
  const files: Record<string, { body: string; type: string }> = {
    [`${BASE}index.html`]: { body: INDEX_HTML, type: 'text/html' },
    [`${BASE}og.png`]: { body: 'PNGDATA', type: 'image/png' },
    [`${BASE}manifest.webmanifest`]: { body: '{"name":"Headroom"}', type: 'application/manifest+json' },
  };
  let online = true;
  const network = async (input: string | Request): Promise<Response> => {
    if (!online) throw new TypeError('Failed to fetch');
    const url = new URL(typeof input === 'string' ? input : input.url, ORIGIN);
    // Like a static host: the bare directory and any query serve index.html.
    const path = url.pathname.endsWith('/') ? `${url.pathname}index.html` : url.pathname;
    const f = files[path];
    return f ? new Response(f.body, { status: 200, headers: { 'content-type': f.type } }) : new Response('not found', { status: 404 });
  };

  const stores = new Map<string, Map<string, Response>>();
  const keyOf = (k: string | Request) => new URL(typeof k === 'string' ? k : k.url, ORIGIN).href;
  const openCache = (name: string) => {
    const store = stores.get(name) ?? new Map<string, Response>();
    stores.set(name, store);
    return {
      put: async (k: string | Request, res: Response) => void store.set(keyOf(k), res.clone()),
      match: async (k: string | Request) => store.get(keyOf(k))?.clone(),
      addAll: async (urls: string[]) => {
        for (const u of urls) store.set(keyOf(u), await network(u));
      },
    };
  };
  const caches = {
    open: async (name: string) => openCache(name),
    keys: async () => [...stores.keys()],
    delete: async (name: string) => stores.delete(name),
    match: async (k: string | Request) => {
      for (const store of stores.values()) {
        const hit = store.get(keyOf(k));
        if (hit) return hit.clone();
      }
      return undefined;
    },
  };

  const listeners: Record<string, Listener[]> = {};
  const self = {
    location: { origin: ORIGIN },
    addEventListener: (type: string, fn: Listener) => (listeners[type] ??= []).push(fn),
    skipWaiting: async () => {},
    clients: { claim: async () => {} },
  };
  const source = serviceWorkerSource({ version: 'test', base: BASE, urls: Object.keys(files) });
  new Function('self', 'caches', 'fetch', source)(self, caches, network);

  const settle = () => new Promise((r) => setTimeout(r, 0));
  return {
    setOnline: (v: boolean) => {
      online = v;
    },
    async install() {
      const waits: Promise<unknown>[] = [];
      for (const fn of listeners.install ?? []) fn({ waitUntil: (p: Promise<unknown>) => waits.push(p) });
      await Promise.all(waits);
    },
    async navigate(path: string): Promise<Response> {
      let responded: Promise<Response> | undefined;
      const request = { method: 'GET', mode: 'navigate', url: `${ORIGIN}${path}` };
      for (const fn of listeners.fetch ?? []) fn({ request, respondWith: (p: Promise<Response>) => (responded = p) });
      if (!responded) throw new Error('service worker did not respond');
      const res = await responded;
      await settle(); // let any fire-and-forget cache writes land
      return res;
    },
  };
}

describe('generated service worker: navigation fallback (#20)', () => {
  it('opening a non-HTML asset in a tab online does not replace the offline app shell', async () => {
    const sw = harness();
    await sw.install();

    // Online: open the manifest and the OG image as top-level navigations.
    expect(await (await sw.navigate(`${BASE}og.png`)).text()).toBe('PNGDATA');
    expect(await (await sw.navigate(`${BASE}manifest.webmanifest`)).text()).toBe('{"name":"Headroom"}');

    // Offline: a never-seen shared link still gets the app shell, not the last asset opened.
    sw.setOnline(false);
    const res = await sw.navigate(`${BASE}?x=1`);
    expect(res.headers.get('content-type')).toBe('text/html');
    expect(await res.text()).toBe(INDEX_HTML);
  });

  it('online navigations are network-first; offline falls back to the precached shell', async () => {
    const sw = harness();
    await sw.install();
    expect(await (await sw.navigate(`${BASE}?m=a`)).text()).toBe(INDEX_HTML);
    sw.setOnline(false);
    expect(await (await sw.navigate(`${BASE}?m=b`)).text()).toBe(INDEX_HTML);
  });
});
