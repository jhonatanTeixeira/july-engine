---
tags:
  - Configuration
  - VRAM
  - MoE
  - MTP
---

# Resource Calculator (VRAM/RAM + MoE)

**File:** `vendor/july_engine_libs/python/llama_gguf/resource_calculator.py` (a separate git submodule from this repo — see [Getting Started](../getting-started.md))

`estimate_vram_ram()` backs two things: the [Admin Panel](admin-panel.md)'s live VRAM estimate (`POST /admin/models/estimate`) and the real pre-load VRAM gate the [orchestrator](../architecture/orchestrator.md) uses before actually loading a GGUF model onto GPU (`GGUF.get_required_vram()`).

## `ModelMetadata`

Reads GGUF header metadata via `gguf.GGUFReader` — both KV fields (`reader.fields`, architecture/layer count/context length/etc.) and, as of the MoE work below, per-tensor byte sizes (`reader.tensors`), cached to disk under `storage/cache/<md5-of-path>_v2.json` so repeated estimates don't re-parse the file. String-type GGUF fields (e.g. `general.architecture`) are raw `uint8` numpy memmaps, not Python `bytes` — decoding them requires `bytes(part).decode("utf-8")`, not `part.tolist()`; this was a real bug fixed as part of the MoE work (previously `architecture` always came back `"unknown"`).

## `vision_on_cpu` — mmproj VRAM accounting

Setting `vision_on_cpu=True` for a vision-capable GGUF model excludes the mmproj (CLIP-style vision projector) file's size from the VRAM estimate, moving it to the RAM total instead — mirroring the real load path, which keeps the vision encoder on CPU when this flag is set.

**Root cause of a "toggling it does nothing" bug, since fixed:** the math was always correct (`mmproj_vram_gb = 0 if vision_on_cpu else meta.mmproj_size_gb`), but `mmproj_size_gb` was always `0` regardless, because `mmproj_repo_id`/`mmproj_filename` were accepted into `estimate_vram_ram()`'s `**kwargs` and never actually read — so there was nothing to subtract either way. The fix resolves `mmproj_path` via `hf_hub_download(repo_id=mmproj_repo_id, filename=mmproj_filename, local_files_only=True)` before constructing `ModelMetadata`, mirroring the real model-load path in `llama_gguf.py`'s `GGUF.load()`. `local_files_only=True` is intentional — a VRAM *estimate* shouldn't trigger a network download just to size-check a file.

The returned dict's `mmproj_vram_mb` is the VRAM-conditional figure (0 when `vision_on_cpu=True`); `mmproj_file_size_mb` is the unconditional file size, so the UI can show "mmproj is 195MB, currently contributing 0 to VRAM" rather than looking like it wasn't detected at all.

## `cpu_moe` / `n_cpu_moe` — Mixture-of-Experts offload

llama-cpp-python exposes two options on `Llama.__init__` for MoE (Mixture-of-Experts) GGUF models:

- **`cpu_moe: bool`** — keep *all* routed-expert weights on CPU, regardless of `n_gpu_layers`.
- **`n_cpu_moe: int`** — keep the routed-expert weights of just the first `N` layers on CPU; useful when VRAM is insufficient for the full model but the CPU has room to spare for some of it.

Both translate internally into `tensor_buft_overrides`: a regex-to-CPU-buffer-type mapping. `cpu_moe=True` uses one global pattern (`\.ffn_(up|down|gate|gate_up)_(ch|)exps`); `n_cpu_moe=N` generates one pattern per layer (`blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps` for `i in range(N)`). Both **exclude** the router (`ffn_gate_inp*`) and shared-expert tensors (`*_shexp*`) — only the large routed-expert blob per layer moves; the router stays on GPU since it's small and needs to run for every token regardless.

### How the calculator accounts for it

`ModelMetadata.expert_count` (0/absent ⇒ dense, not MoE) and `ModelMetadata.moe_expert_bytes(layer_cap)` — which sums `reader.tensors` byte sizes for tensors matching `blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch|)exps` where the captured layer index is `< layer_cap` — are the two building blocks. `estimate_vram_ram()`'s new `cpu_moe`/`n_cpu_moe` parameters compute:

```python
moe_layer_cap = offloaded if cpu_moe else min(n_cpu_moe, offloaded)
moe_offload_bytes = meta.moe_expert_bytes(moe_layer_cap) if (is_moe and (cpu_moe or n_cpu_moe > 0)) else 0
moe_vram_gb = moe_offload_bytes / (1024**3)
weights_vram_gb = max(0, weights_vram_gb - moe_vram_gb)
```

Only layers within the `offloaded` (GPU) range matter here — layers beyond `n_gpu_layers` are already off-GPU in the base weight estimate, so capping at `offloaded` avoids double-counting. Because `weights_vram_gb` is what feeds `total_ram_gb`'s `(file_size_gb - weights_vram_gb)` term, the offloaded MoE bytes automatically show up as extra RAM usage too — no separate addition needed.

The returned dict gains `is_moe`, `expert_count`, and `moe_offload_mb` for UI transparency — the [Admin Panel](admin-panel.md)'s VRAM estimate box shows an extra line ("Especialistas MoE (N) movidos p/ CPU: X MB") only when `is_moe` is true.

On a dense model, `cpu_moe`/`n_cpu_moe` are safe no-ops — `expert_count == 0` short-circuits `moe_offload_bytes` to `0` regardless of the flags.

### End-to-end wiring

Setting these two options isn't just cosmetic to the estimate — they're threaded all the way to the real model load:

1. **Admin Panel** — `model_modal.html`'s advanced accordion (checkbox + number input, `n_cpu_moe` auto-disabled when `cpu_moe` is checked, matching llama-cpp-python's own redundancy warning).
2. **`app/routers/admin_router.py`** form helpers → **`app/routers/models.py`**'s `DownloadRequest`/`UpdateMetadataRequest` → stored in the model catalog.
3. **`vendor/.../llama_gguf.py`**'s `GGUF.__init__` reads `model.get("cpu_moe")`/`model.get("n_cpu_moe")`, and `GGUF.load()`'s `base_params` passes them straight to `Llama(**base_params)` — this is what makes the real model load actually apply the override, not just the pre-load estimate.
4. **`GGUF.get_required_vram()`** and **`app/bridge.py`**'s `process_resource_check()` both pass `cpu_moe`/`n_cpu_moe` (and now, correctly, `mmproj_repo_id`/`mmproj_filename`) through to `estimate_vram_ram()`.

See also [llama-cpp-python Options](llama-cpp-options.md) for how these two fit among every other option this engine configures on the underlying `Llama` object, and [GGUF (llama.cpp) Adapter](../models/gguf_adapter.md) for the model wrapper that consumes all of this.

## `mtp_enabled` — native MTP/NextN speculative decoding

llama.cpp can speculatively decode extra tokens per step using a model's own trained NextN/MTP head(s) — appended-after-the-trunk layer(s) baked into the same GGUF by `convert_hf_to_gguf.py` unless it was run with `--no-nextn`. Unlike `cpu_moe`/`n_cpu_moe`, these tensors are an all-or-nothing load: `llama_model_params.load_mtp` (native default: `False`) gates whether they're read off disk *at all* — there's no partial-offload concept for them.

### How the calculator accounts for MTP

`ModelMetadata.n_layer_nextn` (from the GGUF header's `{arch}.nextn_predict_layers`, 0 ⇒ no MTP layers) and `ModelMetadata.nextn_tensor_bytes()` (sums `reader.tensors` bytes matching `blk\.(\d+)\.nextn\.` or the two top-level `nextn.*_projection` tensors) are the two building blocks — the same shape as `expert_count`/`moe_expert_bytes` above, but simpler since there's no per-layer cap to compute.

```python
has_mtp = meta.n_layer_nextn > 0
mtp_tensor_vram_gb = meta.nextn_tensor_bytes() / (1024**3)
weights_vram_gb = max(0, weights_vram_gb - mtp_tensor_vram_gb)   # pull the proportional-trunk estimate's guess back out...
if mtp_enabled and has_mtp:
    weights_vram_gb += mtp_tensor_vram_gb                        # ...and add the real, full cost back only when it'll actually load
```

The pull-out step runs *unconditionally*: the trunk weight estimate is `file_size_gb * (offloaded / total_layers)`, and NextN blocks sit outside `total_layers`/`block_count` entirely, so that proportional formula always over-counts them slightly regardless of `mtp_enabled`. On top of the tensor bytes, a small **second context** is added when MTP is actually enabled (`mtp_kv_vram_mb`/a small fixed compute-buffer allowance, folded into `mtp_vram_mb` in the returned dict): its KV cache uses the *same* per-position formula as the primary's, just scoped to `n_layer_nextn` layers instead of `total_layers` — confirmed empirically (loading a real MTP GGUF) that it needs one cell per position across the *whole* `n_ctx`, same as the primary, and comes out tiny only because it's 1 layer instead of dozens. Its compute buffer is the opposite story: it must **not** scale with `n_batch`/`n_ctx` the way the primary's does, since the real driver (`Llama._mtp_draft`) never submits more than one row per decode call to this context — the fork's `Llama.__init__` caps this context's own `n_batch`/`n_ubatch` at a small constant regardless of the primary's for exactly this reason (confirmed empirically: reusing the primary's `n_batch` here reserved a ~500MB compute buffer on its own and blew a 4GB GPU's VRAM budget). The RAM total is adjusted the same way as VRAM for the "never even read off disk when disabled" case — see the `resident_file_size_gb` comment in the source.

The returned dict gains `has_mtp`, `n_layer_nextn`, and `mtp_vram_mb` for UI transparency — same idea as `is_moe`/`expert_count`/`moe_offload_mb`, and the [Admin Panel](admin-panel.md)'s MTP toggle only appears when `has_mtp` is true.

### `n_rs_seq` recurrent-state rollback buffer — a real gap, since fixed

The second MTP context's own KV cache/compute buffer (above) is **not** the only extra VRAM cost `load_mtp=True` brings for a hybrid/recurrent architecture. `Llama.__init__` also requests native rollback depth `n_rs_seq = max(n_rs_seq, n_mtp_draft)` on the **primary** context whenever `load_mtp` is on — a small in-VRAM ring buffer (`llama-memory-recurrent.cpp`) that lets a rejected MTP draft be rolled back cheaply for architectures on llama.cpp's `llm_arch_supports_rs_rollback` whitelist (qwen35 included), instead of a slow host checkpoint round-trip. This buffer scales with `n_seq_max` (`n_rows = n_seq_max * (1 + n_rs_seq)`, one row per sequence per rollback slot, across every recurrent layer) and was **missing from the estimate entirely** — for a model with many recurrent layers at a higher `n_seq_max`, it's the single *biggest* MTP-related VRAM cost, dwarfing the second context's own footprint. Missing it is exactly what let `n_seq_max=4 + mtp_enabled=True` silently overflow a real 4GB GPU's budget at context-create time (a raw, unhelpful native CUDA OOM) instead of being caught here, before ever attempting to load.

`ModelMetadata.is_hybrid_recurrent` (true when `ssm.conv_kernel`/`ssm.inner_size`/`ssm.state_size` are all present and non-zero — Mamba-style SSM/recurrent metadata) and `ModelMetadata.n_rs_rollback_bytes(n_seq_max, n_rs_seq)` are the two new building blocks:

```python
n_embd_r = (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)
n_embd_s = ssm_d_state * ssm_d_inner
n_rows = max(1, n_seq_max) * (1 + max(0, n_rs_seq))
bytes = (n_embd_r + n_embd_s) * n_rows * n_recurrent_layers * 4  # f32
```

This mirrors `llama_hparams::n_embd_r()`/`n_embd_s()`'s generic Mamba case (the one qwen35's gated-delta-net layers fall into) and `llama_memory_recurrent`'s own row-count formula — confirmed byte-for-byte against a real loaded model's own logged buffer size (402.00 MiB at `n_seq_max=4`, `n_rs_seq=1`, 24 recurrent layers). `n_recurrent_layers` approximates llama.cpp's per-layer `is_recr(il)` filter via `full_attention_interval` (every Nth layer is full attention instead of recurrent, matching qwen35's hybrid layout); when that metadata is absent, it falls back to treating every layer as recurrent — a safe overestimate, never an underestimate, for a VRAM budget check. `n_rs_seq` itself isn't threaded through `GGUF.load()`'s `base_params` at all, so the estimate assumes `Llama.__init__`'s own default of `1` — matching what actually gets requested in practice today.

Folded into `mtp_vram_mb` and also exposed separately as `mtp_rs_rollback_vram_mb` (and `is_hybrid_recurrent`) in the returned dict, for the same UI-transparency reasons as the other MTP fields. Zero for any non-recurrent architecture — llama.cpp itself silently clamps `n_rs_seq` back to 0 for those regardless of what's requested, so there's nothing to budget for. Regression test: `tests/test_resource_calculator.py::test_mtp_vram_estimate_accounts_for_n_rs_seq_rollback_buffer`.

### MTP end-to-end wiring

Same four stops as `cpu_moe`/`n_cpu_moe` above: `model_modal.html`'s toggle (gated on `has_mtp` via `data-has-mtp`, mirroring `data-is-moe`) → `DownloadRequest`/`UpdateMetadataRequest` → `GGUF.__init__`/`GGUF.load()` (`self.mtp_enabled` → `base_params["load_mtp"]`) → `GGUF.get_required_vram()`/`app/bridge.py`'s `process_resource_check()`, both passing `mtp_enabled` through to `estimate_vram_ram()`.

The actual speculative-decoding driver — drafting via a second `ctx_type=LLAMA_CONTEXT_TYPE_MTP` context sharing the loaded model's weights, verifying against the primary context, and rolling back the KV cache on a rejected draft — lives entirely in the vendored `llama-cpp-python` fork's `Llama` class (`__init__`'s MTP context setup, `_mtp_draft`/`_mtp_verify_after_decode`, and the MTP branch in `generate()`), not in this calculator. This file only ever estimates its footprint.
