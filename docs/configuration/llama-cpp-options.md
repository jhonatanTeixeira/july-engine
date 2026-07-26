---
tags:
  - Configuration
  - Chat
---

# llama-cpp-python Options

July Engine talks to `llama-cpp-python` exclusively through the vendored `vendor/july_engine_libs/python/llama_gguf/llama_gguf.py`'s `GGUF` class (wrapped, in turn, by [`app/models/gguf_adapter.py`](../models/gguf_adapter.md)). `Llama` accepts dozens of constructor parameters; this page covers the subset this engine actually sets, where the value comes from, and what changing it costs you.

## Where values come from

Most of these are resolved in `GGUF.__init__`/`GGUF.load()`'s `base_params` construction, from (in priority order, where applicable): the model's stored metadata (`model_meta`, editable via the [Admin Panel](admin-panel.md)) → an environment variable → a hardcoded engine default.

## Core parameters

| Parameter | Source | Engine default | Notes |
|---|---|---|---|
| `model_path` | `hf_hub_download(repo_id, filename)` | — | Always resolved from the HF Hub cache, never a raw local path passed by the caller. |
| `n_gpu_layers` | `model_meta["num_layers"]` | `-1` (all layers) | `0` forces CPU-only for that model. Values above what fits trigger the [orchestrator](../architecture/orchestrator.md)'s eviction/decrement dance before ever reaching `Llama()`. |
| `n_ctx` | `model_meta["context_window"]` × `n_seq_max` | `4096` (env `LLM_CTX_TOKENS` as fallback) | The engine multiplies the per-request context by `n_seq_max` — VRAM cost scales with the *product*, not either alone. |
| `n_seq_max` | `model_meta["n_seq_max"]`/`["n_parallel"]` | `1` | Parallel request slots on one loaded instance (Sequence Pooling) — each additional slot costs one full extra KV cache, not a fraction. |
| `offload_kqv` | `model_meta["offload_kqv"]` | `True` | When `True`, the KV cache lives in VRAM (faster); when `False`, in RAM (frees VRAM, slower). |
| `kv_unified` | `model_meta["kv_unified"]` | `True` | Unified KV pool — better memory access patterns, at a small VRAM overhead the [resource calculator](resource-calculator.md) models explicitly (~20% of total KV size). |
| `logits_all` | `model_meta["logits_all"]` | `False` | Needed for logprobs/top-k sampling over every token, not just the last — meaningful VRAM overhead when enabled; leave off unless you need it. |
| `flash_attn` | `model_meta["flash_attn"]` → env `FLASH_ATTN` | `True` | ~2x throughput and lower attention-matrix VRAM; on by default. |
| `type_k` / `type_v` | `model_meta["kv_cache_quantization"]` → env `KV_CACHE_QUANTIZATION` | `FP16` | `Q8_0` (~75% smaller KV cache) or `Q4_0` also supported — quantization error is a real quality tradeoff at `Q4_0`. |
| `cpu_moe` / `n_cpu_moe` | `model_meta["cpu_moe"]` / `["n_cpu_moe"]` | `False` / `0` | MoE expert-weight CPU offload — see [Resource Calculator](resource-calculator.md#cpu_moe-n_cpu_moe-mixture-of-experts-offload) for the full mechanics. |
| `use_mmap` | env `USE_MMAP` | `True` | Faster load, lower initial RAM — pages in on demand, which can add latency variance under memory pressure. |
| `n_batch` | env `LLM_N_BATCH` | `max(512, 2048)` | Larger batches improve prompt-processing throughput at the cost of more temporary memory. |
| `n_threads` / `n_threads_batch` | env `MAX_GGUF_THREADS` | CPU thread default | Matters most on CPU-backend models. |
| `verbose` | env `LLM_VERBOSE` | `False` | llama.cpp's own stderr logging — noisy in production. |

## Chat format & tool calling

`chat_format` is resolved from `model_meta["template"]` if set, else auto-detected via `detect_model_capabilities()` against the model identifier (repo id + filename). Two model families get a dedicated custom chat handler instead of a plain `chat_format` string:

- **Qwen** (non-vision) → `QwenChatHandler`, built from the model's own `tokenizer.chat_template` GGUF field (read via `ModelMetadata.tokenizer_template`) — this is what makes Qwen's native tool-calling/reasoning format work correctly.
- **Phi** (non-vision) → `PhiChatHandler`.

## Vision models

When `model_meta["model_type"] == "vision"`, the loader resolves `mmproj_path` via `hf_hub_download(mmproj_id, mmproj_filename)` and picks a chat handler based on the detected vision architecture (`gemma4`, `gemma3`, `qwen3vl`, `qwen25vl`, `qwen35`, `moondream`, `llava-v1.6`/`pixtral`, `llava`) — falling back to `Llava15ChatHandler` if the specific handler can't be imported. `vision_on_cpu` (`model_meta["vision_on_cpu"]`) controls whether the vision encoder itself runs on GPU or CPU (`handler_kwargs["use_gpu"]`) — see [Resource Calculator](resource-calculator.md#vision_on_cpu-mmproj-vram-accounting) for how this is accounted for in the VRAM estimate.

## Options this engine does not currently set

llama-cpp-python exposes many more `Llama.__init__` parameters than the table above — `split_mode`/`main_gpu`/`tensor_split` (multi-GPU), `rope_scaling_type`/`rope_freq_base`/`rope_freq_scale`/YaRN parameters (context extension beyond the model's native length), `draft_model` (speculative decoding), `numa`, `use_mlock`, `check_tensors`, `seed`. None of these are wired to `model_meta` or an env var today — if you need one, follow the same pattern as `cpu_moe`/`n_cpu_moe`: add it to `GGUF.__init__`, thread it into `base_params`, expose it in the [Admin Panel](admin-panel.md)'s advanced accordion, and — if it affects memory footprint — add it to the [resource calculator](resource-calculator.md).

## Recommended presets

**GPU, general chat:**
```
n_gpu_layers=-1, flash_attn=True, offload_kqv=True, kv_unified=True,
logits_all=False, n_seq_max=2-4, n_ctx=4096-8192, kv_cache_quantization=Q8_0 if VRAM-constrained
```

**GPU, MoE model that doesn't fit:**
```
n_cpu_moe=<enough layers to fit>, or cpu_moe=True if GPU headroom is very limited
```
Check the [Admin Panel](admin-panel.md)'s live VRAM estimate while tuning `n_cpu_moe` — it reflects the real reduction per layer.

**CPU-only:**
```
n_gpu_layers=0, use_mmap=True, n_threads=-1 (max), logits_all=False
```
