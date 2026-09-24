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

**Active params** (what decode reads per token, used by the throughput estimate). Dense models read all their params. For MoE models there are two methods, and the Weights card and the MoE note say which one was used:

- **Structural** (used when the config gives `intermediate_size` and `num_attention_heads`), summed over the layer shapes:
  - attention per layer, GQA/MHA: `hidden × (heads × headDim) + 2 × hidden × (kvHeads × headDim) + (heads × headDim) × hidden` (q, k, v, o)
  - attention per layer, MLA (when `kv_lora_rank` and `v_head_dim` exist): `hidden × q_lora_rank + q_lora_rank × heads × (qk_nope + qk_rope) + hidden × (kv_lora_rank + qk_rope) + kv_lora_rank × heads × (qk_nope + v_head_dim) + heads × v_head_dim × hidden`. Without `q_lora_rank` the q term is `hidden × heads × (qk_nope + qk_rope)`.
  - dense FFN for the first `first_k_dense_replace` layers: `3 × hidden × intermediate_size`
  - MoE FFN for the other layers: `3 × hidden × expertFfn × (expertsPerToken + sharedExperts) + hidden × numExperts` (the router), where `expertFfn = moe_intermediate_size ?? intermediate_size`
  - embeddings: `vocab × hidden`, counted once for the LM head. The input embedding is a one-row lookup per token, so it is not read in full each step. That holds whether the embeddings are tied or not.
  - `active = numLayers × attention + denseFfn + moeFfn + vocab × hidden`, capped at the total params. Biases and norms are ignored.
  - Checks against published figures: Qwen3-30B-A3B 3.04B (3.3B), Qwen3-235B-A22B 21.6B (22B), gpt-oss-20b 3.61B (3.6B), gpt-oss-120b 5.13B (5.1B), DeepSeek-V3 36.6B (37B), GLM-4.5-Air 12.8B (12B), Kimi K2 31.7B (32B).
- **Ratio** (the fallback when those fields are missing): `params × (expertsPerToken + sharedExperts) / (numExperts + sharedExperts)`. Attention and embeddings are always active, so this can understate the real figure by a third, which overstates tok/s.

**Fit**
- `effectiveVramGB = vramGB` for non-Apple GPUs; for Apple GPUs (which have unified memory), `effectiveVramGB = min(appleWiredLimitGB, vramGB ≤ 36 ? vramGB × 0.67 : vramGB × 0.75)`. macOS caps GPU-wired memory by default at 0.67× RAM for ≤36 GB, 0.75× above that (observed by MLX and llama.cpp communities). You can raise it with `sudo sysctl iogpu.wired_limit_mb=<MB>`, and Headroom will use that limit if you provide `appleWiredLimitGB` (clamped to vramGB).
- `usable = gpuCount × effectiveVramGB × 1e9 × (1 − reservePct/100)` for the `generic` runtime; other runtimes replace this (see **Runtime profiles**).
- `fixed = weights + overheadGB × 1e9 × gpuCount` (plus a runtime's own overhead allowance, if any).
- `total(N) = fixed + N × kvPerRequest(C)`
- `fits = total(N) ≤ usable`. Headroom under 10% of usable is flagged as **Tight**.
- `maxUsers(C) = floor((usable − fixed) / kvPerRequest(C))`, or 0 if that is negative
- `maxContext(N) = min(maxPositionEmbeddings, floor((usable − fixed) / (N × bytesPerTokenFullAttention)))`. This uses the full-attention rate, so it is conservative: sliding layers only lower it.

**Runtime profiles** (`src/lib/runtime.ts`, `src/lib/launchCommand.ts`). `generic` is the default and reproduces every number above exactly; the others change how usable VRAM (and, for vLLM, KV sizing) is computed, and add a *Launch command* card:

| Runtime | Usable VRAM | Extra overhead | KV sizing |
|---|---|---|---|
| `generic` | `gpuCount × effectiveVramGB × 1e9 × (1 − reservePct/100)` | — | exact context |
| `vllm` | `gpuCount × effectiveVramGB × 1e9 × gpu_memory_utilization` (default 0.9) | + 1 GB × gpuCount, a conservative labelled allowance for CUDA-graph capture and activation buffers | context rounded up to a 16-token block (vLLM's paged KV cache allocates in fixed-size blocks) |
| `sglang` | `gpuCount × effectiveVramGB × 1e9 × mem_fraction_static` (default 0.88) | — | exact context |
| `llamacpp` (covers Ollama) | same as `generic` | — | exact context; `-c` is one pool shared across `-np` parallel slots, so the launch command sets `-c` to `contextTokens × concurrentUsers` and shows/warns on the per-slot context (`c / np`) |
| `mlx` | same as `generic` (Apple unified memory already goes through `effectiveVramGB`'s wired-memory limit above — MLX does not duplicate that math) | — | exact context |

The launch command omits any flag that equals the runtime's own default (e.g. `--tensor-parallel-size` when `gpuCount` is 1, `--kv-cache-dtype`/`--cache-type-k/v` when the chosen KV quant is the runtime's native/unquantized type) and is always labelled a starting point, not a guarantee.

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
