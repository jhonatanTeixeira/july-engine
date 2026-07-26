---
tags:
  - Configuration
  - Admin Panel
---

# Admin Panel

**Router:** `app/routers/admin_router.py` · **Templates:** `app/web/templates/` · **Static assets:** `app/web/static/`

A server-rendered configuration UI at `/admin`, built with Jinja2 + [HTMX](https://htmx.org/) + [Stimulus.js](https://stimulus.hotwired.dev/) — no SPA build step, no separate frontend process. It ported the config UI from the sibling `july_engine_studio` project directly into this engine, calling the existing Python services (`SettingsService`, `ModelsService`, `voice_service`, `bridge`) in-process rather than proxying over HTTP.

!!! note "No authentication"
    The panel currently has no auth layer — treat it as trusted-network-only (e.g. behind your own reverse proxy / VPN), the same way you would the raw `/v1/settings` API.

## Tabs

| Tab | Route (`GET /admin/tab/{tab}`) | Covers |
|---|---|---|
| Geral | `geral` | Global settings — default language, system prompt, internal-MCP toggle. |
| Serviços | `services` | Per-task-family model/backend selection for every non-GGUF task type (`VISION`, `STT`, `TTS`, `IMAGE_EDIT`, `IMAGE_CREATE`, `RESIZE`, `EMBEDDINGS`, `ENTITY_EXTRACTION`, `VIDEO_GENERATION`). |
| Modelos GGUF | `models` | The chat/text model catalog — add/edit/delete GGUF models, live VRAM estimate, advanced llama.cpp options. |
| Chat Presets | `presets` | Named `TEXT_PRESETS` entries (alias, model, backend, default flag). |
| Vozes | `voices` | Uploaded voice-cloning references + YouTube extraction + the built-in Kokoro voice catalog. |
| Monitoramento | `monitoring` | Live RAM/CPU/GPU metrics (polled). |

Each tab is a full-page load on first visit (`GET /admin`) and an HTMX partial swap thereafter (`hx-get="/admin/tab/{tab}" hx-target="#tab-content"`), so switching tabs never reloads the page shell.

## Models tab — add/edit modal

`GET /admin/models/modal/new` (add) / `GET /admin/models/modal/{alias}` (edit) render `partials/model_modal.html`. Key interactions, wired via `model_form_controller.js`:

- Typing a Hugging Face repo ID (`hx-trigger="keyup changed delay:500ms"`) triggers `GET /admin/models/hf-search` (repo autocomplete) and, on filename selection, `GET /admin/models/files` (lists `.gguf` files in that repo).
- Picking a file triggers `POST /admin/models/detect` — real GGUF-header detection (architecture, layer count, context length) plus name-based heuristics (template, vision flag, force-reasoning), auto-filling the form. See [`api_detect_metadata`](../models/gguf_adapter.md) for the underlying implementation, and [Known Issues](../known-issues.md) for its history (it used to be completely broken).
- Every field that affects VRAM (`context_window`, `num_layers`, `kv_cache_quantization`, `mmproj_*`, `vision_on_cpu`, `cpu_moe`, `n_cpu_moe`) triggers `POST /admin/models/estimate`, re-rendering `partials/vram_estimate.html` with the live breakdown from the [resource calculator](resource-calculator.md) — including the MoE-offload line when the loaded model is detected as Mixture-of-Experts.
- The **advanced accordion** exposes `n_seq_max`, `kv_cache_quantization`, GPU layer count, flash attention, KQV offload, KV-unified pool, logits-all, and the two MoE options (`cpu_moe` checkbox auto-disables the `n_cpu_moe` number input, matching llama-cpp-python's own redundancy warning — see [llama-cpp-python Options](llama-cpp-options.md)).
- Clicking **Instalar** (add mode) starts a real-time download: `download_progress_controller.js` opens a streaming `fetch()` (not `EventSource`, since the route is `POST`) against `POST /admin/models/download`, parsing `data: {...}` SSE-style frames off the response body and updating a progress bar through `initializing → starting → downloading → success` stages.

## Voices tab

Three sources of voices, all surfaced in one grid:

1. **Uploaded reference clips** (`POST /admin/voices/upload`) — used by every voice-cloning TTS engine ([Chatterbox](../models/tts_chatterbox.md), [XTTS2](../models/tts_xtts2.md), [Qwen3-TTS](../models/tts_qwen3.md), [NeuTTS Air](../models/tts_neutts_air.md), [F5-TTS](../models/tts_f5.md), [IndexTTS-2](../models/tts_indextts2.md)). `POST /admin/voices/{voice_id}/clean` re-runs the voice-cleaning pipeline (noise reduction/trimming) on an already-uploaded clip.
2. **YouTube extraction** (`POST /admin/voices/youtube/metadata` to preview title/duration/thumbnail, then `POST /admin/voices/youtube/extract` to actually download + register) — via `yt-dlp`, reusing `voice_service.add_voice()`'s existing cleaning pipeline, so a YouTube-sourced clip is processed identically to a manually uploaded one.
3. **Built-in catalog** (`app/services/tts_voice_catalog.py`) — the real 54-voice Kokoro catalog (`af_*`/`am_*`/`bf_*`/etc.), shown read-only for reference since these need no upload/registration to use.

## Monitoring tab

`GET /admin/monitoring/metrics` renders `partials/monitoring_metrics.html` against `app/routers/monitoring.py`'s `get_ram_info`/`get_cpu_info`/`get_gpu_info` — polled on an interval by HTMX (`hx-trigger="every Ns"`) rather than a websocket, since the metrics themselves are cheap to compute and don't need sub-second latency.

## Deliberately not proxied over HTTP

Every admin route calls the same Python services the public API routers call (`SettingsService`, `ModelsService`, `bridge.process_resource_check`, `orchestrator.unload_model`, ...) directly, in-process — there's no internal HTTP round-trip to `/v1/settings` or `/models/gguf` from within `admin_router.py`. One deliberate exception: `model_delete()` does **not** call the existing `/models/gguf` `DELETE` route (which has a real routing bug — see [Known Issues](../known-issues.md)) and instead replicates its body directly (`load_models_db()` → `orchestrator.unload_model()` → delete from dict → `save_models_db()`).
