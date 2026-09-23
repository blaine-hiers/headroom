# Headroom — local LLM VRAM & KV-cache calculator

**Date:** 2026-09-23 · **Appetite:** one session · **Status:** approved (built from the /goal directive; design decisions made by the agent, listed under *Decisions*)

## What it is

A static, single-page GUI (dark mode by default) that answers: *"Will this model fit on my GPU(s), at this context length, for this many concurrent users — and how fast will it decode?"*

Input is a Hugging Face repo id. Headroom reads the model's `config.json` and safetensors parameter count from the Hub, then computes weight memory, KV-cache memory per token / per request, total VRAM for N concurrent users, the maximum users or context that fit, and a bandwidth-bound decode-speed estimate. It is an informational estimator, not a benchmark; every number is labelled as an estimate and the math is shown.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Repo | `blaine-hiers/headroom`, local `Documents\GitHub\Headroom` | Personal tool, same account as Punchcard and limit-pulse |
| Stack | Vite + React 19 + TypeScript, plain CSS with design tokens, Vitest | No backend needed; HF Hub sends CORS headers so the browser can fetch directly. Zero install for users, deploys to GitHub Pages |
| Data source | `GET https://huggingface.co/api/models/{id}?expand[]=safetensors&expand[]=gated` (param count) + `GET https://huggingface.co/{id}/resolve/main/config.json` (architecture) | Verified 2026-09-23: both send `Access-Control-Allow-Origin`; the API's own `config` field is partial, `config.json` is complete |
| Gated models | Optional HF token (stored in `localStorage`, sent as `Authorization: Bearer`) **plus** built-in fallback specs for popular gated models (Llama 3.x, Gemma) | Llama is the model people ask about most and it 401s without a token |
| GGUF repos | Not fetched in v1; user picks the quant manually and enters params by hand or looks up the base BF16 repo | GGUF metadata needs a binary parser; the base repo has the same architecture |
| Theme | Dark default, light toggle, honours `prefers-color-scheme` on first load, choice persisted | Asked for dark mode |
| State | Whole calculator state serialised into the URL query string | Shareable links, no accounts |

## Architecture

```
src/
  lib/                     pure, framework-free, fully unit-tested
    types.ts               ModelSpec, HardwareSpec, Workload, Quant, Result types
    quant.ts               weight + KV quant tables (bits per weight, bytes per KV element)
    kvcache.ts             bytesPerToken(spec, kvQuant), kvBytesForContext(spec, ctx, kvQuant)
    weights.ts             weightBytes(params, quant), activeParams(spec)
    fit.ts                 totalVram, maxUsers, maxContext, fits
    throughput.ts          decode tok/s estimate (bandwidth-bound)
    hf.ts                  fetchModel(id, token?) → ModelSpec; parseConfig(json, params) → ModelSpec
    presets/models.ts      fallback ModelSpecs for gated / popular models
    presets/gpus.ts        GPU table: name, vramGB, bandwidthGBs, vendor
    format.ts              bytes → "320 KB", "2.5 GB"; number formatting
    urlState.ts            encode/decode calculator state ↔ query string
  ui/
    App.tsx                layout: inputs column + results column
    ModelPanel.tsx         HF lookup (id input, fetch, status), preset picker, manual override fields
    QuantPanel.tsx         weight quant + KV quant selectors
    HardwarePanel.tsx      GPU preset, count, custom VRAM/bandwidth, overhead + reserve
    WorkloadPanel.tsx      context length, concurrent users
    Results.tsx            headline cards, context table, fit badge, throughput
    Chart.tsx              VRAM vs concurrent users (inline SVG, no chart lib)
    ShowTheMath.tsx        formula breakdown with the actual numbers substituted
    ThemeToggle.tsx
    theme.css              tokens on :root, dark via [data-theme] and prefers-color-scheme
```

`lib/` has no React imports and is the only place math lives. `ui/` renders `lib` results and never computes.

## Data model

```ts
type Attention = 'mha_gqa' | 'mla';

interface ModelSpec {
  id: string;                 // HF repo id or preset name
  name: string;
  params: number;             // total parameters (from safetensors total)
  activeParams: number;       // = params for dense; MoE estimate otherwise
  numLayers: number;
  attention: Attention;
  numKvHeads: number;         // GQA/MHA
  headDim: number;            // GQA/MHA
  kvLoraRank?: number;        // MLA
  qkRopeHeadDim?: number;     // MLA
  slidingWindow?: number;     // tokens
  slidingLayers?: number;     // how many of numLayers are sliding-window layers
  maxPositionEmbeddings: number;
  hiddenSize: number;
  vocabSize: number;
  nativeDtype: 'bf16' | 'fp16' | 'fp32' | 'fp8';
  moe?: { numExperts: number; expertsPerToken: number; sharedExperts: number };
  source: 'hf' | 'preset' | 'manual';
  warnings: string[];         // e.g. "sliding window ignored by vLLM < 0.6", "MLA: KV is a single latent"
}

interface HardwareSpec { gpuName: string; gpuCount: number; vramGB: number; bandwidthGBs: number; reservePct: number; overheadGB: number }
interface Workload { contextTokens: number; concurrentUsers: number }
interface Quant { weight: WeightQuantKey; kv: KvQuantKey }
```

## Formulas

All sizes in bytes; UI formats with 1000-based GB (labelled) to match vendor VRAM figures, and shows GiB in a tooltip.

**KV bytes per token**
- GQA/MHA: `2 × numLayers × numKvHeads × headDim × kvBytes`
  - `numKvHeads = num_key_value_heads ?? num_attention_heads`
  - `headDim = head_dim ?? hidden_size / num_attention_heads`
- MLA (DeepSeek V2/V3/R1, Kimi K2): `numLayers × (kv_lora_rank + qk_rope_head_dim) × kvBytes` — one compressed latent per token, no ×2.
- Worked check (Llama 3 70B, BF16): `2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 320 KB` ✔

**KV bytes for one request at context C**
- Full-attention layers: `bytesPerTokenPerLayer × C`
- Sliding-window layers: `bytesPerTokenPerLayer × min(C, slidingWindow)`
- Sum across layers. Sliding layer count: `layer_types` array if present (count `"sliding_attention"`); else `sliding_window_pattern` N → `numLayers − ceil(numLayers / N)` sliding; else `use_sliding_window === true` (or `sliding_window` set and model_type is `mistral`) → all layers sliding; else 0.
- `kvBytes`: fp16/bf16 2, fp8 1, int8 1, int4 0.5.

**Weight bytes** = `params × bitsPerWeight / 8`. Table (approximate, llama.cpp effective sizes):

| key | bits | note |
|---|---|---|
| fp32 | 32 | |
| bf16 / fp16 | 16 | native for most |
| fp8 | 8 | native DeepSeek V3, vLLM FP8 |
| q8_0 / int8 | 8.5 | |
| q6_k | 6.56 | |
| q5_k_m | 5.69 | |
| q4_k_m | 4.85 | most common GGUF |
| q4_0 / awq / gptq (g128) | 4.5 | |
| iq4_xs | 4.25 | |
| nf4 (bitsandbytes) | 4.5 | |
| q3_k_m | 3.91 | |
| q2_k | 3.35 | |

**Active params (MoE)** ≈ `params × (expertsPerToken + sharedExperts) / (numExperts + sharedExperts)`, with the note that attention/embedding params are always active so this understates slightly. Dense: `= params`.

**Fit**
- `usable = gpuCount × vramGB × 1e9 × (1 − reservePct/100)`
- `fixed = weights + overheadGB × 1e9 × gpuCount`
- `total(N) = fixed + N × kvPerRequest(C)`
- `fits = total(N) ≤ usable`
- `maxUsers(C) = floor((usable − fixed) / kvPerRequest(C))` (0 if negative)
- `maxContext(N) = min(maxPositionEmbeddings, floor((usable − fixed) / (N × bytesPerTokenFullAttention)))` — uses the full-attention per-token rate (conservative; sliding layers only lower it).

**Decode throughput (estimate, bandwidth-bound)**
- `bytesPerStep(N) = activeWeightBytes + N × kvPerRequest(C)`
- `stepsPerSec = bandwidthGBs × 1e9 × gpuCount × efficiency / bytesPerStep(N)`, `efficiency = 0.7`, multi-GPU assumes tensor parallel so bandwidth adds (labelled)
- per-user tok/s = `stepsPerSec`; aggregate tok/s = `stepsPerSec × N`.
- Prefill is not estimated (compute-bound; out of scope).

## HF parsing rules (`parseConfig`)

1. If `text_config` exists (Llama 4, Gemma 3, Qwen-VL, Mistral 3 multimodal), read architecture fields from it, falling back to top level.
2. `model_type` in `{deepseek_v2, deepseek_v3, kimi_k2}` or `kv_lora_rank` present → `attention: 'mla'`.
3. MoE if any of `num_local_experts`, `n_routed_experts`, `num_experts` > 1; `expertsPerToken = num_experts_per_tok ?? 1`; `sharedExperts = n_shared_experts ?? 0`.
4. `params` from the API's `safetensors.total`; if missing (no safetensors, e.g. pickle-only or GGUF repo) → `warnings` gets "parameter count not available — enter manually" and the field is left editable.
5. 401/403 → "gated, private, or misspelled: check the id, add an HF token, or pick the built-in preset" (verified: the Hub returns 401 for non-existent repos too); 404 → "repo not found"; network failure → plain message. Never throws to the UI: returns `{ ok: false, error }`.
6. `warnings` collects: sliding-window present, MLA, MoE active-params estimate, fp8 native weights (quant table starts at fp8), `params` missing.

## UI

Two columns on desktop (inputs 380px | results fill), single column under 800px, 16px gutters, no horizontal scroll.

- **Model panel**: text input `org/model`, *Fetch* button, quick-pick chips (Llama 3.1 8B, Llama 3.3 70B, Qwen3-32B, Qwen3-30B-A3B, Gemma 3 27B, gpt-oss-120b, DeepSeek-V3.1, Mistral Small 3.2). Status line (fetching / fetched from HF / preset / error). *Advanced* disclosure exposes every ModelSpec field for manual override; edits switch `source` to `manual`. HF token field lives here, masked, with a "stored in this browser only" note.
- **Quant panel**: weight quant select (table above, native dtype pre-selected), KV cache dtype select.
- **Hardware panel**: GPU preset select (grouped NVIDIA consumer / NVIDIA datacenter / AMD / Apple / other), count stepper, editable VRAM and bandwidth, reserve % (default 5), runtime overhead GB per GPU (default 1.0).
- **Workload panel**: context slider + number (256 … maxPositionEmbeddings, log scale), concurrent users number (1 … 512).
- **Results**:
  - Headline cards: *KV per token*, *Weights*, *KV for N users @ C*, *Total VRAM*, with a **Fits / Does not fit** badge and the headroom in GB.
  - Table: 2K / 8K / 32K / 128K (and the chosen C if different) → KV per request, max users at that context.
  - *Max context for N users* and *Max users at C* callouts.
  - Throughput card: per-user and aggregate tok/s, labelled "bandwidth-bound estimate, ×0.7 efficiency".
  - Chart: total VRAM vs concurrent users (1 … 2×maxUsers or 64), horizontal line at usable VRAM, inline SVG, uses theme tokens.
  - *Show the math*: the formulas above with the live numbers substituted, in a `<details>`.
  - Warnings list from `ModelSpec.warnings`.
- **Header**: name, one-line description, theme toggle, *Copy link* (URL state), GitHub link.

Everything recomputes synchronously on any input change; no submit button except *Fetch*.

## Error handling

- All HF failures surface in the model panel status line; the calculator keeps its last good spec.
- Non-numeric or out-of-range inputs clamp to bounds; nothing renders NaN (format() renders "—" for non-finite).
- URL state that fails to parse is ignored and defaults load.

## Testing

- `lib/` ≥ 90% covered by Vitest: the Llama 3 70B worked example is a golden test. The slide's figures are binary: 327,680 B/token = 320 KiB (328 KB decimal), 640 MiB @ 2K, 2.5 GiB @ 8K, 10 GiB @ 32K, 40 GiB @ 128K (42.9 GB), 400 GiB for 10 users @ 128K; weights 70.6 B × 2 = 141 GB. Tests assert exact bytes plus both unit strings. Plus GQA vs MHA, MLA (DeepSeek-V3 config fixture), sliding window (Gemma 3 fixture with `sliding_window_pattern`, gpt-oss with `layer_types`), MoE active-params, quant table, fit/maxUsers/maxContext boundaries, URL round-trip, `parseConfig` on real config.json fixtures saved under `src/lib/__fixtures__/`.
- `hf.ts` tested with a mocked `fetch` for 200 / 401 / 404 / network error.
- UI: one smoke test that App renders and the golden numbers appear for the Llama 3.3 70B preset.
- `npm run build` and `npm run lint` clean.

## Deployment

GitHub Actions workflow builds on push to `main` and deploys to GitHub Pages; `vite.config.ts` sets `base: '/headroom/'`. README carries the URL, a screenshot, the formulas, and the "estimates, not benchmarks" caveat.

## Out of scope (v1)

GGUF metadata parsing, prefill/TTFT estimation, speculative decoding, tensor-parallel communication overhead, CPU offload split, multiple models at once, account/login.
