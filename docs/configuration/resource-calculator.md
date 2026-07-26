---
tags:
  - Configuration
  - VRAM
  - MoE
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
