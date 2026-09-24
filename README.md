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

**Weight bytes from repo files.** For a GGUF repo, and for a pre-quantized safetensors repo (a `quantization_config` in `config.json`: AWQ, GPTQ, FP8…), Headroom uses the exact size of the weight files instead of the estimate:

- GGUF: the size of the chosen `.gguf` file, or of all its shards added up for a split file (`-00001-of-0000N`).
- Safetensors: the sum of the repo's top-level `*.safetensors` files (a Mistral-style duplicate `consolidated.safetensors` is left out).

The file figure only applies while the weight quant is the one the files are in. Loading the repo pre-selects that quant. bitsandbytes repos map to NF4 (`load_in_4bit`) or INT8 (`load_in_8bit`). GGUF tags not in the table (IQ2_M, Q5_K_S…) and other methods get the nearest table quant by effective bits (`fileBytes × 8 / params`). When the parameter count is unknown, the GGUF tag family decides instead (IQ2_M → Q2_K, Q4_K_XL → Q4_K_M…), and the Weights card names that closest table quant. Files that match no table quant are not used: a model note says so and the weights are estimated. Pick any other quant and the estimate comes back. An edit under **Advanced** to the parameters or the architecture also drops the file figure. The Weights card says which source it used: *from repo files* with the effective bits per weight, or *estimated*. For decode, the active share of the files is `fileBytes × activeParams / params`.

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
- `usable = gpuCount × effectiveVramGB × 1e9 × (1 − reservePct/100)`
- `fixed = weights + overheadGB × 1e9 × gpuCount`
- `total(N) = fixed + N × kvPerRequest(C)`
- `fits = total(N) ≤ usable`. Headroom under 10% of usable is flagged as **Tight**.
- `maxUsers(C) = floor((usable − fixed) / kvPerRequest(C))`, or 0 if that is negative
- `maxContext(N) = min(maxPositionEmbeddings, floor((usable − fixed) / (N × bytesPerTokenFullAttention)))`. This uses the full-attention rate, so it is conservative: sliding layers only lower it.

**Multi-GPU (tensor parallel)**. With more than one GPU, Headroom assumes tensor parallelism (VRAM and bandwidth add across GPUs). vLLM and most runtimes refuse to start when the head count can't be split evenly, so Headroom checks it and warns next to the fit badge:
- `numAttentionHeads % gpuCount === 0` (from the config's `num_attention_heads`, when known). If not, the nearest GPU counts that do divide evenly are suggested. Models without that field (no structural `ffn` spec) skip the check rather than warn falsely.
- `numKvHeads < gpuCount` (GQA/MHA only — see below) forces KV-head replication (some GPUs hold a duplicate KV head to keep the split even). Headroom models this: `kvReplicationFactor = gpuCount / numKvHeads` multiplies the KV total (KV per token, per request, and the context table), and a warning explains the resulting per-GPU size increase. The layout is only even when `numKvHeads % gpuCount === 0` (split) or `gpuCount % numKvHeads === 0` (replicate) — e.g. 3 KV heads over 8 GPUs satisfies neither, so that pairing is flagged too, with suggested GPU counts that satisfy both the head-count check and this one.
- **MLA models** (DeepSeek V2/V3/R1, Kimi K2) cache one compressed KV latent per token with no per-head KV projections, so KV-head replication and its split-validity check never apply to them — only the attention-head divisibility check does.

**Decode throughput** (an estimate: decode is limited by memory bandwidth)
- `bytesPerStep(N) = activeWeightBytes + N × kvPerRequest(C)`
- `stepsPerSec = bandwidthGBs × 1e9 × gpuCount × efficiency / bytesPerStep(N)`. With more than one GPU this assumes tensor parallelism, so bandwidth adds up — but real tensor-parallel communication (all-reduce between GPUs each step) is not otherwise modeled, so `efficiency` carries a small labelled scaling penalty for it: `efficiency = 0.7 × 0.9^log2(gpuCount)`, i.e. ×0.9 for each doubling of GPU count (1 at a single GPU, so single-GPU numbers are unchanged).
- per-user tok/s = `stepsPerSec`; aggregate tok/s = `stepsPerSec × N`

**Prefill / time-to-first-token** (an estimate: prefill is compute-bound, not bandwidth-bound)
- `prefillFlops ≈ 2 × activeParams × promptTokens + 2 × numLayers × promptTokens² × queryWidth`. The first term is the matmuls (same accounting as decode); the second is the O(n²) attention term (QKᵀ and attention × V).
  - `queryWidth = numAttentionHeads × headDim` when the config gives a head count (`ffn.numAttentionHeads`); otherwise `hiddenSize` stands in, since `numAttentionHeads × headDim ≈ hiddenSize` for the query projection. Which one was used is labelled next to the estimate.
- `TTFT = prefillFlops / (tflopsBf16 × 1e12 × gpuCount × MFU)`, with `MFU` (model FLOPS utilization) defaulting to **0.4**, labelled as such. `tflopsBf16` is the GPU's dense (no-sparsity) BF16 tensor rate; the estimate keeps that rate even at lower weight quants (an fp8 rate would need a separate GPU field).
- Shown for one user at the chosen context, next to the decode throughput, as a compute-bound estimate.

## Where the model data comes from

For a repo id, Headroom makes three requests straight from your browser. The Hub sends CORS headers, so no proxy is needed.

- `GET https://huggingface.co/api/models/{id}?expand[]=safetensors&expand[]=gated`, which gives the parameter count (`safetensors.total`)
- `GET https://huggingface.co/{id}/resolve/main/config.json`, which gives the architecture: layers, heads, head dim, MLA, sliding window, MoE and max positions
- `GET https://huggingface.co/api/models/{id}?blobs=true&expand[]=siblings&expand[]=gguf`, which lists the repo's files with their sizes, and for GGUF repos the parameter count (`gguf.total`)

**Searching for a repo id.** As you type in the repo id field, Headroom debounces ~250ms and calls `GET https://huggingface.co/api/models?search=<query>&limit=10&sort=downloads&expand[]=gated`, which also sends CORS headers. The dropdown always lists matching bundled presets first (so it works offline), then Hub results with their download count and a **gated** badge where the API reports one. A failed search (offline, rate-limited) is silent — the preset matches still work.

**GGUF repos** (for example `bartowski/…-GGUF`) have no `config.json`. When a repo has `.gguf` files and no `.safetensors` files, Headroom lists the GGUF files, grouping split shards, and shows a **GGUF file** picker. It defaults to Q4_K_M when there is one. The quant comes from the filename, and the file's size is the weight size. The architecture comes from the file's own header: Headroom reads the first 1 MiB with an HTTP `Range` request (`bytes=0-1048575`) on `resolve/main/{file}`. The Hub redirects that to its CDN, which answers ranged CORS reads. From the header's metadata it reads `general.architecture`, then `<arch>.block_count`, `.attention.head_count`, `.attention.head_count_kv`, `.attention.key_length`, `.context_length`, `.embedding_length`, `.feed_forward_length`, `.expert_count`, `.expert_used_count`, `.expert_shared_count`, `.expert_feed_forward_length`, `.leading_dense_block_count`, `.attention.sliding_window` and the MLA keys (`.attention.kv_lora_rank`, `.attention.q_lora_rank`, `.rope.dimension_count`, `.attention.key_length_mla`, `.attention.value_length_mla`). The vocabulary size is the length of `tokenizer.ggml.tokens`. These go through the same rules as `config.json`. GGUF files don't store which layers slide, so Headroom uses llama.cpp's fixed patterns: every 2nd layer is full attention for Gemma 2 and gpt-oss, every 4th for Cohere 2, and every 6th for Gemma 3. For other architectures no layers slide, which is the conservative choice.

If the listing or the header read fails, you get a status message and the calculator keeps its last model. A 401/403 on the header read shows the gated-repo hint. If the repo also has a usable `config.json`, Headroom uses it, as it did before GGUF support, with estimated weights and a status note. You can then enter the architecture under **Advanced**, or look up the base BF16 repo, which has the same architecture.

Gated repos such as Llama and Gemma return 401 without a token. You have two options:

- Paste a Hugging Face token under **Gated models**. It is stored only in your browser's `localStorage` and sent only to huggingface.co.
- Use a built-in preset.

## Adding a GPU or model preset

Presets are plain data in [`src/lib/presets/`](src/lib/presets/):

- **GPU:** add an entry to `GPU_PRESETS` in `gpus.ts` with its `name`, `vendor` (`nvidia-consumer`, `nvidia-datacenter`, `amd`, `apple` or `other`), per-GPU `vramGB` in decimal GB, `bandwidthGBs`, and `tflopsBf16` (dense, no-sparsity BF16 tensor TFLOPS from the vendor spec sheet; used for the TTFT estimate). It appears in the GPU select under its vendor group, with all three fields editable once selected.
- **Model:** add a `preset({...})` entry to `MODEL_PRESETS` in `models.ts`. Take the values from the repo's `config.json` and the Hub API's `safetensors.total`. It appears as a quick-pick chip. Add `warnings` for anything a user should know, such as sliding windows, MoE or pre-quantized weights.

Then run `npm test`: `presets.test.ts` checks every preset for sane values.
