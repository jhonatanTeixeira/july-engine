---
tags:
  - API
---

# API Reference

Every endpoint is served by a router that calls `bridge.process_*` and nothing else — see [Request Flow](architecture/request-flow.md) for how a call actually reaches a model. Routers are vendored (`july_routers`, shared with other services in the Jully ecosystem) except where noted as engine-specific.

## Control headers

| Header | Effect |
|---|---|
| `x-backend: gpu\|cpu` | Pre-selects the backend context; derived from model settings if omitted. |
| `x-context-window` | Per-request context-window override (GGUF chat). |
| `x-session-id` | KV-cache sequence-slot affinity across turns of the same conversation. |
| `x-nsfw: 1` | [FLUX.2 Klein](models/flux_klein.md)-specific — enables an optional LoRA if present. |
| `x-max-steps` | [FLUX.2 Klein](models/flux_klein.md)-specific — overrides `num_inference_steps` for image generation/editing. Default `4`; raise it for higher quality at the cost of latency. |

## Chat (OpenAI-compatible)

```
POST /v1/openai/chat/completions
POST /v1/openai/embeddings
POST /v1/openai/audio/speech
POST /v1/openai/audio/transcriptions
POST /v1/openai/images/edits
POST /v1/openai/images/generations
POST /v1/openai/images/resize
```

## Chat (Anthropic-compatible)

```
POST /v1/anthropic/messages
POST /v1/anthropic/embeddings
POST /v1/anthropic/audio/speech
POST /v1/anthropic/audio/transcriptions
POST /v1/anthropic/images/generations
POST /v1/anthropic/images/edits
```

`Bridge.process_anthropic_message` translates the Anthropic `system` field into an OpenAI-style system message and, for non-streaming responses, converts the OpenAI-shaped result back into Anthropic's response schema.

## Vision & video (native `/july/v1` schema)

```
POST /july/v1/vision/images/describe          # FastVLM / Moondream* / Molmo
POST /july/v1/vision/video/describe           # Frame-sampled video description
POST /july/v1/vision/images/remove-background # rembg
POST /july/v1/vision/faces/extract            # Face detection + embedding
POST /july/v1/vision/face/sync                # Batch face matching
POST /july/v1/vision/face/embedding
```
*See [Known Issues](known-issues.md) — the `moondream` engine is currently non-functional.

## RAG / memory (native `/july/v1` schema)

```
POST   /july/v1/rag          # add
POST   /july/v1/rag/batch    # batch add
POST   /july/v1/rag/search
POST   /july/v1/rag/vector
PUT    /july/v1/rag/update
DELETE /july/v1/rag/{item_id}
POST   /july/v1/rag/batch-delete
GET    /july/v1/rag/list
POST   /july/v1/rag/smart-search   # search + LLM summarization
```

See [Persistence & Vector Store](architecture/persistence.md) for the `RAG_DATABASE` backend this reads/writes.

## Entity extraction

```
POST /july/v1/entities/extract
```

See [GLiNER2](models/gliner2_extractor.md).

## Video generation

!!! warning "Implemented models, no HTTP route"
    `Bridge.process_video_generation` (task type `video_generation`) exists and is fully wired through the [orchestrator](architecture/orchestrator.md)/[`ImageAdapter`](models/wan2_t2v.md) down to real, working model wrappers — [Wan2.2 T2V](models/wan2_t2v.md), [Wan2.2 I2V](models/wan2_i2v.md), and [LTX-2](models/ltx2_video.md) — but **no router in this repo currently exposes an HTTP endpoint that calls it**. Today the only way to exercise these models is directly in Python (each has a `if __name__ == "__main__":` standalone CLI) or by adding a route yourself that calls `bridge.process_video_generation(payload, headers)`, following the same one-line pattern as `process_image_generation`/`process_image_edit` in `app/bridge.py`.

## Utility

```
POST /july/v1/utils/extract-pdf
POST /july/v1/check-resources   # VRAM/RAM pre-load estimate — see Resource Calculator
```

## Models, settings & voices (engine-specific, `app/routers/`)

```
GET    /models/gguf                        # list registered GGUF models
POST   /models/gguf/download                # SSE-style download progress stream
POST   /models/gguf/detect_metadata         # heuristics + real GGUF header detection
GET    /models/gguf/files/{repo_id}         # list .gguf files in a HF repo
PUT    /models/gguf/{alias}                 # update stored model metadata
DELETE /models/gguf/gguf/{alias}            # NOTE: double "gguf" segment — see below
POST   /models/gguf/warmup                  # pre-load models by task_type/alias

GET  /v1/settings
POST /v1/settings

GET    /voices/
POST   /voices/upload
PATCH  /voices/{voice_id}
DELETE /voices/{voice_id}
POST   /voices/{voice_id}/clean

GET /monitoring/                            # RAM/CPU/GPU snapshot
GET /health
GET /metrics                                # Prometheus scrape endpoint
```

!!! note "The delete route's path really does have a doubled segment"
    `app/routers/models.py`'s router is mounted with `prefix="/models/gguf"`, and its delete route is itself declared as `@router.delete("/gguf/{model_alias}")` — so the full path is `/models/gguf/gguf/{model_alias}`, not the `/models/gguf/{model_alias}` you'd expect by symmetry with the other routes on this router. The [Admin Panel](configuration/admin-panel.md) deliberately doesn't call this route (see its own docs) precisely to sidestep this.

## Admin Panel

See [Admin Panel](configuration/admin-panel.md) for the full `/admin/*` route list — it's a server-rendered HTMX/Stimulus UI, not a JSON API, and calls the same underlying services as the routes above rather than proxying through them.
