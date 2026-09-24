#!/usr/bin/env node
// Re-fetches each ungated MODEL_CATALOG entry's config.json + Hub safetensors total and
// diffs the resulting ModelSpec against what is bundled in src/lib/presets/catalog.ts, using
// the exact same parseConfig() the live "fetch a repo" path uses. Gated repos (Meta, Google)
// 401 without a token and are reported as skipped, not fetched.
//
// Usage: node scripts/check-catalog.mjs
// Exits 1 on drift, an unexpected fetch error, or a skipped id whose org isn't recognized as
// gated, so a CI run or a reviewer can tell "no drift" from "didn't actually check".

import { register } from 'node:module';

register(new URL('./ts-loader.mjs', import.meta.url));

const { MODEL_CATALOG } = await import('../src/lib/presets/catalog.ts');
const { parseConfig } = await import('../src/lib/hf.ts');

// Orgs whose repos 401 without a Hub token; their entries are hand-verified instead of re-fetched.
const GATED_ORGS = new Set(['meta-llama', 'google']);

const FIELDS = [
  'params',
  'numLayers',
  'attention',
  'numKvHeads',
  'headDim',
  'kvLoraRank',
  'qkRopeHeadDim',
  'slidingWindow',
  'slidingLayers',
  'maxPositionEmbeddings',
  'hiddenSize',
  'vocabSize',
  'nativeDtype',
];

function diffField(field, bundled, fresh) {
  const a = bundled[field];
  const b = fresh[field];
  if (a === b) return undefined;
  if (a === undefined && b === 0) return undefined; // absent vs. explicit 0 (e.g. slidingLayers)
  return `${field}: bundled=${JSON.stringify(a)} fresh=${JSON.stringify(b)}`;
}

// Key order isn't meaningful for moe/ffn (object literal order vs. the order parseConfig
// assigns fields in), so compare with sorted keys rather than raw JSON.stringify.
function stableJson(v) {
  return JSON.stringify(v, Object.keys(v ?? {}).sort());
}

function diffJson(field, bundled, fresh) {
  const a = stableJson(bundled[field]);
  const b = stableJson(fresh[field]);
  if (a === b) return undefined;
  return `${field}: bundled=${JSON.stringify(bundled[field])} fresh=${JSON.stringify(fresh[field])}`;
}

async function fetchFresh(id) {
  const path = id.split('/').map(encodeURIComponent).join('/');
  const apiUrl = `https://huggingface.co/api/models/${path}?expand[]=safetensors`;
  const configUrl = `https://huggingface.co/${path}/resolve/main/config.json`;
  const [apiRes, configRes] = await Promise.all([fetch(apiUrl), fetch(configUrl)]);
  if (!configRes.ok) throw new Error(`config.json HTTP ${configRes.status}`);
  const config = await configRes.json();
  let params;
  if (apiRes.ok) {
    const api = await apiRes.json();
    params = api?.safetensors?.total;
  }
  return parseConfig(config, params, id);
}

let drift = 0;
let checked = 0;
let skipped = 0;
let errors = 0;

for (const { spec } of MODEL_CATALOG) {
  const org = spec.id.split('/')[0];
  if (GATED_ORGS.has(org)) {
    console.log(`SKIP   ${spec.id} (gated on the Hub — hand-verified, not re-fetched)`);
    skipped++;
    continue;
  }
  try {
    const fresh = await fetchFresh(spec.id);
    const diffs = [
      ...FIELDS.map((f) => diffField(f, spec, fresh)),
      diffJson('moe', spec, fresh),
      diffJson('ffn', spec, fresh),
    ].filter(Boolean);
    if (diffs.length > 0) {
      console.log(`DRIFT  ${spec.id}`);
      for (const d of diffs) console.log(`         ${d}`);
      drift++;
    } else {
      console.log(`OK     ${spec.id}`);
    }
    checked++;
  } catch (e) {
    console.log(`ERROR  ${spec.id}: ${e instanceof Error ? e.message : e}`);
    errors++;
  }
}

console.log('');
console.log(`${checked} checked, ${drift} drifted, ${skipped} gated (skipped), ${errors} errors`);

if (drift > 0 || errors > 0) process.exit(1);
