---
tags:
  - Architecture
---

# Persistence & Vector Store

There are **two independent** config/storage layers in this engine, selected by two different environment variables — don't confuse them.

## Settings & model catalog

`app/persistence/base.py` defines `PersistenceBackend` (abstract): settings by key, the GGUF model catalog, uploaded voices, MCPs, and history events. `app/persistence/persistence.py`'s `get_backend()` picks the concrete implementation via `PERSISTENCE_BACKEND`:

| Value | Backend | Storage |
|---|---|---|
| `tinydb` (default) | `TinyDBBackend` | `storage/db/tinydb.json` (path overridable via `DB_PATH`) |
| `postgres` | `PostgresBackend` | `DATABASE_URL` connection string |

If `PostgresBackend` fails to connect, `get_backend()` raises a `RuntimeError` with a clearer message rather than silently falling back — a broken `DATABASE_URL` fails loudly at startup, not with a confusing downstream error later.

Application code should go through the service layer rather than `get_backend()` directly:

- `app/services/settings_service.py` — read/write settings by key (`TEXT_PRESETS`, `VISION`, `TTS`, `STT`, `EMBEDDINGS`, `IMAGE_EDIT`, `IMAGE_CREATE`, `RESIZE`, `BG_REMOVAL`, `ENTITY_EXTRACTION`, `VIDEO_GENERATION`, ...), exposed over HTTP at `/v1/settings`.
- `app/services/models_service.py` — the GGUF model catalog (used by `ModelLoader` to resolve `model_meta` for a given alias), exposed at `/v1/models`.

There is no static `settings.yaml` in this repo, despite older documentation implying one — all configuration is dynamic, DB-backed key/value, editable at runtime through the [Admin Panel](../configuration/admin-panel.md) or the settings API directly.

## RAG vector store

`app/persistence/vector_store.py`'s `VectorStore` is a completely separate Strategy Pattern over embedding storage, selected via `RAG_DATABASE`:

| Value | Backend |
|---|---|
| `in-memory` (default) | Plain in-process store — lost on restart |
| `chroma` | ChromaDB |
| `pgvector` | PostgreSQL + pgvector extension |

This is used exclusively by `RagAdapter`/`app/adapters/rag_adapter.py` for the memory/RAG endpoints (`/july/v1/rag/*`) and has no relationship to `PERSISTENCE_BACKEND` — you can run `PERSISTENCE_BACKEND=postgres` with `RAG_DATABASE=chroma`, or any other combination, independently.

## Why they're kept separate

Settings/model-catalog data is small, relational, and changes rarely (a handful of KB of JSON/rows). Embeddings are potentially large, need similarity search rather than key lookup, and benefit from a purpose-built vector index. Coupling them to the same backend/schema would force every deployment into the more complex vector-capable option even when RAG isn't used at all — keeping them independent lets a minimal deployment run `tinydb` + `in-memory` with zero external services.
