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

Any model registered under `TEXT_PRESETS` settings — there's no alias/engine branching here at all, since GGUF is the only text-generation backend. Configure per-model options (`context_window`, `n_seq_max`, `kv_cache_quantization`, `flash_attn`, `offload_kqv`, `kv_unified`, `logits_all`, `vision_on_cpu`, `cpu_moe`, `n_cpu_moe`, `mtp_enabled`, mmproj repo/filename for vision-capable GGUF models) via the [Admin Panel](../configuration/admin-panel.md) or directly through `/models/gguf`.

`mtp_enabled` turns on native MTP/NextN speculative decoding for GGUFs that were converted with their trained NextN head(s) included (the default for `convert_hf_to_gguf.py` unless `--no-nextn` was passed) — the admin form only shows the toggle when the resource estimator's `has_mtp` comes back true for the selected file. Under the hood this sets `load_mtp=True` on the vendored `llama_cpp.Llama` (so the NextN tensors actually get read off disk) and it builds a second, small context sharing the same loaded weights to draft and verify extra tokens per step — see the vendored fork's `Llama.__init__`/`generate()` MTP driver for the mechanics.

Verified end-to-end against `unsloth/Qwen3.5-4B-MTP-GGUF`: greedy-decoding output matched the non-MTP baseline token-for-token except one accepted GPU floating-point near-tie flip (batched vs. single-token GEMMs aren't bit-identical — an expected, industry-standard characteristic of speculative decoding, not a bug), and throughput measured **~1.15x over the non-MTP baseline** at the default `n_mtp_draft=1` on the target hardware (RTX 3050 4GB). Chain-heads (multiple distinct trained NextN layers, `n_layer_nextn > 1`) is implemented but unverified against a real multi-head model.

Qwen3.5 is a **hybrid** architecture (recurrent/linear-attention "gated delta net" layers mixed with attention). Rejecting a draft for a hybrid model can't be rolled back with a plain KV-position delete the way a normal attention-only model's rejection can — but it doesn't need a slow, generic full-state checkpoint either: llama.cpp has a purpose-built, in-VRAM mechanism for exactly this (`n_rs_seq`, a small ring buffer of recent recurrent-state snapshots kept in device memory — see `llama-memory-recurrent.cpp`/`llama-memory-hybrid[-iswa].cpp`), and `qwen35` is on its own `llm_arch_supports_rs_rollback` whitelist. `Llama.__init__` requests `n_rs_seq` depth equal to `n_mtp_draft` whenever `load_mtp` is on, and the *same* `memory_seq_rm` call already used for plain attention models transparently does the right thing for hybrid models too — no separate code path needed. (An earlier version of this implementation didn't know that mechanism existed, used a generic host-round-trip checkpoint cache instead, and measured a net *slowdown* as a result — the checkpoint overhead alone outweighed the win. That's fixed; the numbers above are post-fix.)

**Draft depth (`n_mtp_draft`) defaults to 1**, not deeper: for this single-head model, recurrently chaining the one trained head into itself for additional speculative tokens was measured to *lose* throughput past depth 1 — the head's own uncertainty compounds fast (accept rate dropped from ~90% at depth 1 to ~35% by depth 3), burning compute on speculative work that mostly gets rejected. Raise it only if you've confirmed a specific model's accept rate holds up deeper.

### Concurrency: bugs found and fixed while validating MTP alongside multi-seq batching

MTP was explicitly required to keep working correctly *alongside* the existing concurrent multi-seq continuous batching (`SeqAllocator` + the fork's `DecodeGate`), not just for a single conversation at a time. Validating that surfaced several real, pre-existing concurrency bugs in the vendored fork — none specific to MTP itself (all reproduce with `mtp_enabled=False` too), but all directly relevant to any deployment where more than one conversation can share a loaded model:

- **`HybridCheckpointCache` (`llama_cache.py`) had zero thread-safety.** `_reuse_prefix_and_eval` calls into it (`save_checkpoint`/`find_best_checkpoint`/`restore_checkpoint`) whenever a seq_id's new prompt misses the KV-cache prefix match — an ordinary event, not a rare one, for a hybrid/recurrent architecture like qwen35's gated-delta-net layers. Two concurrent conversations hitting this at once, with no lock at all around the shared checkpoint list or the native `llama_state_seq_*_ext` calls, reliably produced a **"double free or corruption" heap-corruption crash** within a handful of concurrent rounds. Fixed with a dedicated `threading.Lock` around every method that touches the shared checkpoint list or calls those native functions.
- **Four call sites used unscoped "clear everything" fallbacks** (`_hybrid_cache_mgr.clear()` + `self._ctx.memory_clear(True)`) where only the CURRENT seq_id's state should have been invalidated. Under concurrent multi-seq load this silently discarded — or wiped the KV cache of — every *other* active conversation on the same model, producing output that was perfectly coherent but belonged to a completely different conversation. Fixed with a new `HybridCheckpointCache.clear_seq(seq_id)` and scoped `memory_seq_rm(seq_id, 0, -1)` calls in place of the unscoped pair.
- **The hybrid "N-1 checkpoint" prompt-prefill branch in `generate()` didn't sample atomically.** Its final prompt token was decoded but never sampled via `sampling_ctx` (unlike the plain single-token path), so the first generated token silently fell through to a later, lock-free `idx=-1` read of the shared logits buffer — which another concurrently-decoding seq_id could overwrite first. Manifested as a single spurious extra token at the very start of a round (e.g. an inserted "was" before an otherwise-correct continuation). Fixed by sampling atomically in the same `decode_gate`-locked call, exactly like the non-hybrid path already did.
- **`DecodeGate.submit_tokens` didn't restore `active_seq_id` on the way out.** When a caller's thread ends up "leading" a batch that also includes another thread's pending request, `_drain_and_decode` (correctly) flips the `active_seq_id` context var to each request's own seq_id while processing them in turn — but left it pointing at whichever was processed *last*, not necessarily the original caller's own seq_id. The very next per-sequence property access on that thread (`n_tokens`, `input_ids`, `_sampling_ctx`, …) then silently read or wrote a *different* sequence's state. Fixed by unconditionally restoring `active_seq_id` to the caller's own seq_id before `submit_tokens` returns.
- **MTP's own `self._mtp_ctx`/`self._mtp_batch` (a single shared context and a capacity-1 batch across every seq_id) had a lock defined (`self._mtp_lock`) but never actually acquired anywhere.** Two seq_ids drafting concurrently raced on the shared batch, surfacing as `LlamaBatch overflow[add_token_embedding]: capacity 1 reached`. Fixed by acquiring `self._mtp_lock` around every touch of `self._mtp_ctx`/`self._mtp_batch`, including the ones that already run inside `decode_gate`'s own (different, non-overlapping) lock.
- **Intermittent, unrecoverable CUDA aborts** (~1-in-4 concurrent stress runs) with no diagnostic message reaching Python (ggml's own abort path bypasses the logging callback). Traced to CUDA graph capture/replay under a *varying* batch shape — exactly what `DecodeGate` produces by coalescing a changing number of sequences' single-token steps into one `llama_decode()` call per round. `GGML_CUDA_DISABLE_GRAPHS=1` (now set by default at `llama_cpp` import time, in `llama_cpp/__init__.py`) eliminated it across repeated stress runs afterward, for a measured ~1-2% throughput cost — negligible for this fork's actual workload of small, few-token decode steps rather than the large graphs CUDA graphs are meant to amortize launch overhead for.

Regression coverage for the class of bug this represents (crash *or* silently-wrong-conversation output under real concurrent multi-seq chat) lives in july-engine's own suite: `tests/test_integration.py::test_concurrent_hybrid_model_no_corruption_or_crash`, driven by `tests/_hybrid_concurrency_worker.py`. It runs in a subprocess deliberately — a real heap-corruption abort or CUDA abort takes down the whole interpreter, which would otherwise silently kill the rest of the pytest run instead of failing just this one test.

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
