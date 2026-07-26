---
tags:
  - Known Issues
---

# Known Issues & Dead Paths

An honest inventory of code that exists in this repo but doesn't work, or doesn't run at all, so you don't waste time debugging around it or copy its patterns into new code.

## Fully dead / unregistered code

These files are never imported by `main.py` — none of their endpoints are reachable via HTTP in this app, regardless of any bugs inside them.

- **`app/routers/models_router.py`** (prefix `/v1/models`) — entirely unregistered; superseded by `app/routers/models.py` (prefix `/models/gguf`) and `app/services/models_service.py`. Two of its endpoints (`create_or_update_model`, `delete_model`) also have a function-local `from ..orchestrators.gpu_orchestrator import gpu_orchestrator` that would raise `ModuleNotFoundError` even if the router were ever wired in — that package doesn't exist; the real orchestrator is the single `app/orchestrator.py` singleton. Don't copy this import pattern, and don't register this router without first fixing it.
- **`vendor/july_engine_libs/python/routers/july_routers/search.py`** (web search / GitHub search / scraping) — not imported in `main.py` at all. Confirms the README's statement that external search moved to the separate **July Search** service; the routes still exist in the vendored library (shared code) but this app never mounts them.
- **`app/adapters/vision_adapter.py`, `image_adapter.py`, `tts_adapter.py`** still reference a `..services.llm_api` module and an `"api"` engine/backend branch that doesn't exist in this repo — leftovers from when July Engine could proxy to external providers. `app/model_loader._ADAPTER_REGISTRY`'s `"search_adapter"` entry was removed in the same cleanup; treat any remaining `backend == "api"` branches you find as dead, not a pattern to extend.

## Broken model-adapter wiring (reachable, but always fail)

These routes *are* reachable, but the adapter's call into the model wrapper doesn't match that wrapper's actual signature — every request against them fails today.

| Alias / engine | Adapter | Failure |
|---|---|---|
| `moondream` (vision) | `VisionAdapter` | [Moondream](models/moondream.md) — constructor doesn't accept `model_meta`; even past that, a hard-coded `raise` in its load method, missing `is_loaded()`, and a `NameError` in `unload()`. Completely non-functional. |
| `emotion` (vision) | `VisionAdapter` | [Emotion Detection](models/emotion.md) — adapter calls `model.run({"image": img})` (a dict); the model's `run()` expects a raw `PIL.Image` — `AttributeError`. |
| `tagger` (vision) | `VisionAdapter` | [Tagger](models/tagger.md) — adapter calls `model.run(...)`; the model has no `run()` method at all (only `tag()`) — `AttributeError`. |
| `pix2pix` (image edit) | `ImageAdapter` | [InstructPix2Pix](models/pix2pix.md) — adapter imports a class name (`Pix2PixPipeline`) that doesn't exist in the file (the real class is `Pix2Pix`); caught by a broad `except`, silently resolves to no strategy. |
| `video` (video generation) | `ImageAdapter` | [Stable Diffusion Video (legacy)](models/stable_diffusion_video.md) — adapter passes a `use_sequential_offload` kwarg the constructor doesn't accept — `TypeError`, caught the same way. Superseded by [Wan2.2](models/wan2_t2v.md)/[LTX-2](models/ltx2_video.md), which work correctly. |
| `detect_metadata` → admin `/admin/models/detect` | `app/routers/models.py` | **Fixed** as of the `cpu_moe`/`n_cpu_moe` work — previously imported a nonexistent module and called nonexistent attributes; now returns real GGUF header data. Mentioned here for history/context. |

## Known behavioral bugs (reachable, run without error, wrong result)

- **[Multilingual E5](models/multilingual_e5.md)** — the query/passage prefix is inferred from which payload key is present, not from the caller's explicit `emb_type` — in practice this means the `"query: "` E5 prefix is never actually used, even for RAG search queries, which can measurably hurt retrieval quality.
- **[BGE Micro](models/bge_micro.md)** — if `hf_hub_download` fails during a GPU-provider retry, the exception handler references a variable that may not be defined yet, raising `NameError` instead of falling back to CPU cleanly.
- **[BERT / CodeBERT / GraphCodeBERT](models/bert.md)** — never calls `super().__init__()` (so `model_meta` is inert for this wrapper), has no `unload()` at all, and its `run_passage()`/`run_query()` convenience methods pass a raw string into a method that expects a dict — calling either raises `AttributeError`.
- **[FastVLM](models/fastvlm.md)** — multi-image requests silently drop every image but the first (the adapter's batch-detection check looks for a public `run_batch` method; this class only exposes a private `_run_batch`). `run()` is also synchronous and blocks the event loop, since the `async def chat()` wrapper that would offload it via a thread is never actually called by `VisionAdapter`.

## Documented incomplete features (not bugs — deliberate scaffolding)

- **[Trellis2](models/trellis2.md)** — a deliberate stub; `run()` always raises `NotImplementedError` until the model's SDNQ card documents a real usage pattern.
- **[`OnnxUpscalerModel`](models/image_resizer.md#onnxupscalermodel-documented-no-op-skeleton)** — looks for an ONNX super-resolution weight file, but even when one loads successfully, always falls back to the plain Pillow+unsharp-mask upscaler. No default ONNX SR model ships with the project yet.

## VRAM accounting gaps

`app/models/image_resizer.py`'s `ResizerBase` has no `get_required_vram()` method at all — every resize/restoration strategy, including the real GPU models ([Real-ESRGAN, GFPGAN, CodeFormer](models/image_resizer.md)), is reported as costing `0` VRAM to the orchestrator. `ImageAdapter._resize()` compensates partially by force-unloading the heavy strategies immediately after each run, but there's no pre-load VRAM gate for them the way there is for GGUF/SDNQ models — a request against one of these on a nearly-full GPU can still OOM rather than being cleanly rejected or queued.
