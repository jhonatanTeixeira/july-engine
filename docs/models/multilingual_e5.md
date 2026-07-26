---
tags:
  - Embeddings
  - CPU
  - GPU
---

# Multilingual E5

**File:** `app/models/multilingual_e5.py` · **Class:** `MultilingualE5Model(BaseModel)`

Wraps `sentence_transformers.SentenceTransformer` for `intfloat/multilingual-e5-small`, producing E5-style prefixed (`"query: "` / `"passage: "`) normalized embeddings.

## Selecting it

Task type `embeddings`, settings key `EMBEDDINGS`. `RagAdapter._get_emb_model()` matches when the configured model id normalizes to `multilingual-e5` (both `multilingual-e5` and `multilingual_e5` work, since `-`/`_` are normalized before comparison).

## Request payload

```json
{"input": "texto em português para gerar embedding"}
```

`run()` is synchronous, returns a flat `List[float]` (`SentenceTransformer.encode(..., normalize_embeddings=True)`).

## VRAM

`get_required_vram()`: `0` on CPU, `500` MB on GPU.

!!! bug "Known issue — query/passage prefix selection"
    E5 models are trained expecting a `"query: "` prefix for search queries and `"passage: "` for indexed documents — using the wrong one measurably hurts retrieval quality. This wrapper decides which prefix to use via `emb_type = "passage" if payload.get("input") or payload.get("text") else "query"` — i.e. it infers the type from **which payload key is present**, not from an explicit `emb_type` value. `RagAdapter._embed_text` always calls it with `payload={"input": text, "emb_type": emb_type}`, so the `"input"` key is always present and `emb_type` always resolves to `"passage"` — even when the caller's actual intent (e.g. embedding a search query in `rag_search`) was `"query"`. The explicit `emb_type` value passed by the adapter is silently ignored by this model. This looks like a real bug affecting RAG search quality, not just a style issue.
