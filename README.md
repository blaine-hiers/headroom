# Headroom

**Will it fit? For how many? How fast?**

Headroom is a local-LLM VRAM and KV-cache calculator. Give it a Hugging Face repo id (or pick a built-in preset), a quantization, your GPU(s), a context length and a number of concurrent users. It works out:

- weight memory at the chosen quant
- KV cache per token and per request, including sliding-window layers
- total VRAM for N concurrent users, compared with usable VRAM
- the most users that fit at your context, and the longest context that fits for your users
- a bandwidth-bound decode-speed estimate

It runs entirely in the browser. There is no backend and no account. The whole calculator state lives in the URL, so you can share a link to it.

**Live:** https://blaine-hiers.github.io/headroom/

![screenshot](docs/screenshot.png)

> **Estimates, not benchmarks.** Real runtimes (vLLM, llama.cpp, TGI, MLX…) add activation memory, allocator fragmentation, CUDA graphs and their own KV block rounding. Use Headroom to size hardware and settings, then confirm on the real stack. The *Show the math* panel shows every formula with your numbers filled in.

## Run it

```sh
npm i && npm run dev
```

| Command | What it does |
|---|---|
| `npm run dev` | dev server at `http://localhost:5173/headroom/` |
| `npm test` | Vitest: the math layer in node, the UI in jsdom |
| `npm run lint` | oxlint |
| `npm run build` | type-check and production build into `dist/` |

## Units

Every byte figure shows two numbers. The decimal one comes first (1 GB = 10⁹ B), because GPU vendors quote VRAM that way: a "24 GB" RTX 4090 has 24 × 10⁹ bytes. The binary one follows in smaller text (1 GiB = 2³⁰ B), because most LLM slides and runtime logs use binary units. That is why Llama 3 70B's KV cache per token reads **328 KB · 320 KiB**. Both are the same 327,680 bytes.

## Formulas

All sizes are in bytes.

**KV bytes per token**
- GQA/MHA: `2 × numLayers × numKvHeads × headDim × kvBytes`
  - `numKvHeads = num_key_value_heads ?? num_attention_heads`
  - `headDim = head_dim ?? hidden_size / num_attention_heads`
- MLA (DeepSeek V2/V3/R1, Kimi K2): `numLayers × (kv_lora_rank + qk_rope_head_dim) × kvBytes`. MLA stores one compressed latent per token, so there is no ×2.
- Worked check (Llama 3 70B, BF16): `2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 320 KiB`

**KV bytes for one request at context C**
- Full-attention layers: `bytesPerTokenPerLayer × C`
- Sliding-window layers: `bytesPerTokenPerLayer × min(C, slidingWindow)`
- The request total is the sum across all layers. The sliding layer count comes from the first of these that the config has:
  1. a `layer_types` array: count the `"sliding_attention"` entries
  2. `sliding_window_pattern` N: `numLayers − ceil(numLayers / N)` layers slide
  3. `use_sliding_window === true`, or `sliding_window` set on a `mistral` model: every layer slides

  If none of them is present, no layers slide.
- `kvBytes`: fp16/bf16 2, fp8 1, int8 1, int4 0.5.

**Weight bytes** = `params × bitsPerWeight / 8`. The bits per weight are approximate llama.cpp effective sizes, scales included:

| quant | bits/weight |
|---|---|
| fp32 | 32 |
| bf16 / fp16 | 16 |
| fp8 | 8 |
| q8_0 / int8 | 8.5 |
| q6_k | 6.56 |
| q5_k_m | 5.69 |
| q4_k_m | 4.85 |
| q4_0 / awq / gptq (g128) | 4.5 |
| iq4_xs | 4.25 |
| nf4 (bitsandbytes) | 4.5 |
| q3_k_m | 3.91 |
| q2_k | 3.35 |

**Active params (MoE)** ≈ `params × (expertsPerToken + sharedExperts) / (numExperts + sharedExperts)`. Attention and embedding weights are always active, so this slightly understates the real figure. For dense models the active params are all the params.

**Fit**
- `usable = gpuCount × vramGB × 1e9 × (1 − reservePct/100)`
- `fixed = weights + overheadGB × 1e9 × gpuCount`
- `total(N) = fixed + N × kvPerRequest(C)`
- `fits = total(N) ≤ usable`. Headroom under 10% of usable is flagged as **Tight**.
- `maxUsers(C) = floor((usable − fixed) / kvPerRequest(C))`, or 0 if that is negative
- `maxContext(N) = min(maxPositionEmbeddings, floor((usable − fixed) / (N × bytesPerTokenFullAttention)))`. This uses the full-attention rate, so it is conservative: sliding layers only lower it.

**Decode throughput** (an estimate: decode is limited by memory bandwidth)
- `bytesPerStep(N) = activeWeightBytes + N × kvPerRequest(C)`
- `stepsPerSec = bandwidthGBs × 1e9 × gpuCount × 0.7 / bytesPerStep(N)`. With more than one GPU this assumes tensor parallelism, so bandwidth adds up.
- per-user tok/s = `stepsPerSec`; aggregate tok/s = `stepsPerSec × N`
- Prefill (time to first token) is compute-bound and is not estimated.

## Where the model data comes from

For a repo id, Headroom makes two requests straight from your browser. The Hub sends CORS headers, so no proxy is needed.

- `GET https://huggingface.co/api/models/{id}?expand[]=safetensors&expand[]=gated`, which gives the parameter count (`safetensors.total`)
- `GET https://huggingface.co/{id}/resolve/main/config.json`, which gives the architecture: layers, heads, head dim, MLA, sliding window, MoE and max positions

Gated repos such as Llama and Gemma return 401 without a token. You have two options:

- Paste a Hugging Face token under **Gated models**. It is stored only in your browser's `localStorage` and sent only to huggingface.co.
- Use a built-in preset.

GGUF repos are not parsed. Look up the base BF16 repo instead, since it has the same architecture, then choose the GGUF quant.

## Adding a GPU or model preset

Presets are plain data in [`src/lib/presets/`](src/lib/presets/):

- **GPU:** add an entry to `GPU_PRESETS` in `gpus.ts` with its `name`, `vendor` (`nvidia-consumer`, `nvidia-datacenter`, `amd`, `apple` or `other`), per-GPU `vramGB` in decimal GB, and `bandwidthGBs`. It appears in the GPU select under its vendor group.
- **Model:** add a `preset({...})` entry to `MODEL_PRESETS` in `models.ts`. Take the values from the repo's `config.json` and the Hub API's `safetensors.total`. It appears as a quick-pick chip. Add `warnings` for anything a user should know, such as sliding windows, MoE or pre-quantized weights.

Then run `npm test`: `presets.test.ts` checks every preset for sane values.
