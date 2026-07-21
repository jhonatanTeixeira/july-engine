# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

July Engine (`july_engine`) is the local inference core of the "Jully" ecosystem: a FastAPI multimodal engine that serves chat (OpenAI/Anthropic-compatible), vision, TTS/STT, image generation/editing, and RAG entirely with local models (GGUF via llama-cpp-python, PyTorch/Transformers, Diffusers, Coqui/Kokoro/Piper TTS, DeepFace/OpenCV). It auto-manages VRAM/RAM (loads, keeps models warm, LRU-evicts, decrements GPU layers under pressure). External web/code search used to live here but has moved to a separate **July Search** service — this repo no longer calls external LLM APIs in the normal path (see "Known dead paths" below).

Repo comments, logs, and docstrings are frequently in Portuguese; match that style in `app/` if you add comments there.

## Setup & Running

Submodules must be initialized before installing — `vendor/july_engine_libs` provides the `july_routers`, `july_telemetry`, and `llama_gguf` packages (installed editable), and `vendor/IP-Adapter` is also installed editable:

```bash
git submodule update --init --recursive
```

Full environment setup (compiles `llama-cpp-python` with the right backend flags, then `pip install -r requirements.txt`):

```bash
./setup.sh                    # CUDA build for RTX 3050 (default, CUDA_ARCH=86)
WITH_VULKAN=true ./setup.sh   # AMD/Intel via Vulkan
CPU_ONLY=true ./setup.sh      # No GPU acceleration
RECOMPILE=true ./setup.sh     # Force recompiling the llama-cpp-python wheel
CUDA_ARCH=89 ./setup.sh       # Override target CUDA arch
SETUP_UBUNTU=true ./setup.sh  # Also install Tesseract OCR (pt+eng)
```

All dependencies live in the single `requirements.txt` (there is no CPU/GPU split despite what `AGENTS.md` says — see "AGENTS.md vs reality" below).

Run the dev server:

```bash
uvicorn main:app --host 0.0.0.0 --port 3000 --reload   # matches main.py's __main__
```

Docker: `Dockerfile` (prod image, port 8000, runs `./setup.sh` at build time) and `Dockerfile.dev` (dev image with `--reload`, also port 8000). Note the port mismatch with the bare `uvicorn` command above — 3000 locally, 8000 in both Docker images.

Observability stack (Prometheus + Grafana, scraping `/metrics`):

```bash
docker compose -f docker-compose.grafana.yml up -d
# Grafana:    http://localhost:3001 (admin / july)
# Prometheus: http://localhost:9090
```

`.env` (or `.env.<ENV>` when the `ENV` env var is set) is loaded at startup by `main.py`.

## Testing

```bash
pytest                                          # full suite
pytest tests/test_integration.py::test_health_check   # single test
pytest --cpu-only / --gpu-only / --api-only     # filter by marker (see tests/conftest.py)
pytest -m cpu                                   # same, via marker directly
```

`tests/test_integration.py` boots the real FastAPI app in-process (`httpx.ASGITransport`) against the actual `bridge`/persistence backend — no mocking of the inference stack; tests `pytest.skip()` themselves when a local model isn't available rather than mocking it. `tests/test_cloud_path.py` is the exception: it unit-tests `app/services/storage/cloud_path.py` with `unittest.mock` against a fake `fsspec` filesystem, which is fine since the "no mocks" convention is specifically about the model/inference pipeline, not storage I/O.

## Architecture

### Request flow

```
july_routers (vendor/july_engine_libs, e.g. openai.py/anthropic.py/july.py)
  -> app/bridge.py: Bridge (single instance, "dumb" routing — no domain logic)
  -> app/orchestrator.py: Orchestrator.submit_task(task_type, payload)
  -> Runner (resolves model via ModelLoader, manages VRAM/RAM + warm cache)
  -> app/adapters/*: task-level Adapter (Strategy Pattern dispatch)
  -> app/models/*: concrete low-level wrapper around the actual library
```

`main.py` wires everything: it imports routers from the vendored `july_routers` package and calls each module's `set_bridge(bridge)` to inject the single `Bridge` instance (`app/bridge.py`). Routers never talk to the orchestrator directly — they only call `bridge.process_*` methods.

### Bridge (`app/bridge.py`)

Pure routing: injects request headers into the payload and calls `orchestrator.submit_task(task_type, payload)`. `Bridge._TASK_TO_SETTING` maps each `task_type` (e.g. `"text_chat"`, `"tts"`, `"image_edit"`) to the persistence setting key that holds its model config (e.g. `"TEXT_PRESETS"`, `"TTS"`, `"IMAGE_EDIT"`). Also handles OpenAI↔Anthropic payload translation for `process_anthropic_message`.

### Orchestrator / Runner / Context (`app/orchestrator.py`)

One `Orchestrator` singleton holds exactly two shared context singletons — `GpuContext` and `CpuContext` (`Orchestrator._backend_contexts`) — so model "loaded/idle/busy" state is consistent across every request on that backend, regardless of which model. For each task, a `Runner` is created that:
1. Resolves the model instance via `model_loader.get(task_type, backend, model_tag)`.
2. Computes required VRAM/RAM (`get_required_vram`) and compares to `resource_manager.get_available_*_mb()`.
3. If insufficient, evicts the least-recently-used **idle** model on that context (`unload_next`), then tries `decrement_layers()` on GGUF models, then waits, before raising `MemoryError`.
4. Loads the model if not already loaded, runs it, then marks it idle again — **models stay warm/loaded after use**; they're only unloaded to make room for something else or via explicit `/v1/models/{alias}` delete.

`x-backend: cpu|gpu` request header (read in `Orchestrator.submit_task`) pre-selects the context; if absent, `Runner.__init__` derives the backend from the resolved model's own settings.

### ModelLoader (`app/model_loader.py`)

Caches Adapter instances by `f"{task_type}_{resolved_backend}_{model_tag}"`. `_ADAPTER_REGISTRY` maps `task_type` → a lazy getter for the Adapter class (`ChatAdapter`, `VisionAdapter`, `TTSAdapter`, `STTAdapter`, `RagAdapter`, `ImageAdapter`). Each Adapter class declares its own settings key via `get_engine_type(task_type)`.

### Adapters (`app/adapters/*`) — the real Strategy Pattern layer

Each Adapter is one per task family and does its own internal engine/model resolution from the model alias, then instantiates the matching low-level wrapper from `app/models/`:
- `ChatAdapter._get_strategy()` → `GGUFAdapter` (cpu/gpu, wraps `llama_gguf.GGUF`) or `LLMAdapter` (non-local).
- `VisionAdapter._get_vision_model()` → `FastVLMModel` / `MoondreamModel` / `EmotionModel` / `TaggerModel`, resolved from `meta["alias"]`/`meta["model"]`; also handles video description and face extraction/sync.
- `TTSAdapter._get_tts_model()` → `KokoroTTSModel` / `ChatterboxTTSModel` / `XTTS2Model` / `PiperModel` / `FasterQwen3TTSModel`, resolved via `_ALIAS_ENGINE_MAP` prefix matching on the model alias.
- `ImageAdapter._get_strategy()` → `BgRemoverModel` (rembg), one of the resizer models in `app/models/image_resizer.py` (Pillow/OpenCV/GFPGAN/CodeFormer/RealESRGAN/etc.), `Pix2PixPipeline`, `LCMFaceIDPipeline`, or `FluxKleinPipeline`.

**To add a new model wrapper**: implement it in `app/models/` as a `BaseModel` subclass (`load`, `unload`, `is_loaded`, `get_required_vram`, `run`), then wire it into the relevant Adapter's resolution method by alias/tag. Only touch `Bridge._TASK_TO_SETTING` / `model_loader._ADAPTER_REGISTRY` if you're introducing a genuinely new `task_type`, not a new model within an existing one. You never need to touch `orchestrator.py` — it's generic over all task types.

### Persistence vs VectorStore — two independent config/storage layers

- **Settings & model catalog**: `app/persistence/base.py` defines `PersistenceBackend` (settings by key, model catalog, uploaded voices, MCPs, history events). `app/persistence/persistence.py:get_backend()` picks the implementation via `PERSISTENCE_BACKEND` env var: `tinydb` (default, `storage/db/tinydb.json`) or `postgres` (needs `DATABASE_URL`). Read/write task settings through `app/services/settings_service.py` / `app/services/models_service.py`, exposed over HTTP at `/v1/settings` and `/v1/models`. There is no static `settings.yaml` in this repo despite the README section of that name — all config is dynamic, DB-backed key/value.
- **RAG embeddings**: `app/persistence/vector_store.py` is a separate Strategy Pattern over `chroma` / `pgvector` / `in-memory`, selected via the `RAG_DATABASE` env var — unrelated to `PERSISTENCE_BACKEND`.

### Lazy imports for heavy ML libraries

Heavy libraries (`torch`, `diffusers`, `faster_whisper`, `transformers`, `insightface`, `cv2`, etc.) are imported inside the function/method that uses them, not at module top level, throughout `app/`, so the server boots without pulling every ML dependency into memory. Follow this when adding new model wrappers.

### DTO terminology

Request payloads select behavior via a `model` field (e.g. `model: "gfpgan"`, `model: "qwen3-cpu"`) — not `engine`/`tool`/`backend`. `backend` is reserved for the `x-backend` HTTP header (cpu/gpu) and internal routing, not a payload field.

## Known dead / inconsistent paths (don't extend these)

- `app/routers/models_router.py` imports `from ..orchestrators.gpu_orchestrator import gpu_orchestrator` in `create_or_update_model` and `delete_model` — that package doesn't exist (only the single `app/orchestrator.py` with one `orchestrator` singleton). These two endpoints currently raise `ModuleNotFoundError` at runtime; don't copy this import pattern.
- Several adapters (`vision_adapter.py`, `image_adapter.py`, `tts_adapter.py`) still reference `..services.llm_api` and an `"api"` engine/backend branch, and `model_loader._ADAPTER_REGISTRY` still has a `_get_search_adapter()` entry for `app/adapters/search_adapter.py` — neither `services/llm_api.py` nor `adapters/search_adapter.py` exist in this repo. These are leftovers from when July Engine could proxy to external providers/search; per the README, that responsibility moved to the separate July Search service and the `x-backend: api` path was removed. Treat these branches as non-functional, not as a pattern to build on.
- `AGENTS.md` describes an idealized architecture (`july_engine/engine_models/`, `july_engine/domain/`, `july_engine/orchestrators/`, split `requirements_cpu.txt`/`requirements_gpu.txt`, a `_get_strategy()` convention on "Domain" classes) that doesn't match this source tree's actual paths. Trust the structure documented above over AGENTS.md's directory names. Its behavioral rules that *do* hold here — lazy ML imports, `model`-not-`engine` in DTOs, VRAM cleanup via `resource_manager.clear_memory()`, Strategy Pattern per adapter — are folded into this file.

## Licensing

MIT (`LICENSE`), but several vendored/optional dependencies carry more restrictive licenses (AGPL-3.0, GPL-3.0) and some model weights are non-commercial — check `NOTICE.md` before commercial or network-service use.
