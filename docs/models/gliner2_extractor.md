---
tags:
  - Entity Extraction
  - CPU
  - GPU
---

# GLiNER2

**File:** `app/models/gliner2_extractor.py` · **Class:** `GLiNER2Extractor(BaseModel)`

Zero-shot structured entity extraction via [GLiNER2](https://github.com/fastino-ai/GLiNER2) (`fastino/gliner2-base-v1`) — **not** the classic GLiNER library used in the sibling `dating_v3` project; GLiNER2 is a distinct, newer library with a different API surface (`extract_entities`/`batch_extract_entities` rather than GLiNER's `predict_entities`).

## Endpoint

`POST /july/v1/entities/extract` (`app/routers/entities_router.py`) → `Bridge.process_entity_extraction` → `EntityAdapter` → this model.

## Selecting it / configuring labels

Task type `entity_extraction`, settings key `ENTITY_EXTRACTION`. If a request doesn't specify `labels`, `EntityAdapter` falls back to a project-wide default label set (`_DEFAULT_LABELS` in `app/adapters/entity_adapter.py`, matching the sibling `dating_v3` project's list) rather than requiring every caller to enumerate entity types.

## Loading

`quantize` defaults to `True` on CPU backend and `False` on GPU (no extra dependency required either way per the library) unless explicitly overridden via `model_meta["quantize"]`.

## VRAM

`get_required_vram()`: `0` on CPU, `800` MB on GPU.

## Request payload

```json
{
  "text": "John works at Acme Corp in New York.",
  "labels": ["person", "organization", "location"],
  "threshold": 0.5,
  "include_confidence": true,
  "include_spans": true
}
```

- `text` may be a single string or a list of strings — a list is dispatched to `batch_extract_entities` (with `batch_size`, default 8) instead of `extract_entities`.
- `labels` is **required** (`ValueError` if empty) — the list of entity types to look for; there's no fixed schema, any string works as a label.
- `threshold` (default `0.0`, meaning no filtering): GLiNER2's `extract_entities`/`batch_extract_entities` have no native confidence-threshold parameter (that only exists on the unrelated `extract_json` method), so filtering is done here as a manual post-processing step (`_filter_by_threshold`) over the returned per-label entity lists.

## Response shape

```json
{"entities": {"person": [{"text": "John", "confidence": 0.98, "start": 0, "end": 4}], "organization": [...], "location": [...]}}
```

Labels with no surviving entities (after threshold filtering) are dropped from the response entirely, not returned as empty lists.
