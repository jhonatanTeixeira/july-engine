---
tags:
  - Chat
  - CPU
  - GPU
---

# GGUF (llama.cpp) Adapter

**File:** `app/models/gguf_adapter.py` · **Class:** `GGUFAdapter(BaseModel)`

The engine's only chat/text-generation backend. A thin `BaseModel`-conforming shim around the **vendored** `llama_gguf.GGUF` class (`vendor/july_engine_libs/python/llama_gguf/llama_gguf.py`, installed editable) — the real llama.cpp/GGUF inference engine. Every method here just delegates to `self._impl`; the substantial logic (KV-cache sequence allocation, context-window overflow handling, tool-calling, chat-format/vision-handler auto-detection) lives in the vendored package, not in this file.

!!! note "Not the same as ChatAdapter"
    `app/adapters/chat_adapter.py`'s `ChatAdapter` is the higher-level orchestration layer (MCP/tool-call/vision preprocessing) — it always resolves to this `GGUFAdapter` as its only strategy (the dead `"api"` backend branch that used to exist here was removed; see [Known Issues](../known-issues.md)).

## Selecting it

Any model registered under `TEXT_PRESETS` settings — there's no alias/engine branching here at all, since GGUF is the only text-generation backend. Configure per-model options (`context_window`, `n_seq_max`, `kv_cache_quantization`, `flash_attn`, `offload_kqv`, `kv_unified`, `logits_all`, `vision_on_cpu`, `cpu_moe`, `n_cpu_moe`, mmproj repo/filename for vision-capable GGUF models) via the [Admin Panel](../configuration/admin-panel.md) or directly through `/models/gguf`.

## VRAM

`get_required_vram()` delegates to the vendored `GGUF.get_required_vram()`, which calls the shared [resource calculator](../configuration/resource-calculator.md) (`estimate_vram_ram()`) — not a hardcoded constant like most other wrappers in this directory, since GGUF models vary enormously in size.

## Loading & layer offload

`load(n_ctx=None, num_layers=None)` delegates to `GGUF.load(...)`. Also exposes `decrement_layers()`, used by the [orchestrator](../architecture/orchestrator.md) to progressively push GPU layers to CPU when VRAM is tight, before giving up with `MemoryError`.

## Request payload

```json
{"messages": [{"role": "user", "content": "Hello!"}], "stream": false}
```

`run()` pops `messages`/`stream` out of the payload and calls the vendored `GGUF.run_chat(messages, stream=stream, **kwargs)`, which additionally understands the `x-session-id` header (KV-cache slot affinity across an agentic loop's turns) and `x-context-window` header (per-request context override), remaps `repetition_penalty`→`repeat_penalty` and `max_completion_tokens`→`max_tokens`, and forces `repeat_penalty=1.0` in JSON response-format mode.

## Related documentation

- [Resource Calculator (VRAM/RAM + MoE)](../configuration/resource-calculator.md) — the VRAM/RAM estimator this model uses, including the `vision_on_cpu` fix and new `cpu_moe`/`n_cpu_moe` options.
- [llama-cpp-python Options](../configuration/llama-cpp-options.md) — a full parameter-by-parameter reference of what this engine actually configures on the underlying `Llama` object.
