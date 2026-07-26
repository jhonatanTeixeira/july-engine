---
tags:
  - Embeddings
  - CPU
  - GPU
---

# BERT (CodeBERT / GraphCodeBERT)

**File:** `app/models/bert.py` · **Classes:** `BertEmbedder` (base), `CodeBERT`, `CodeGraphBERT`

Base class wrapping `transformers.AutoModel`/`AutoTokenizer` (BERT-family) for mean-pooled, L2-normalized sentence embeddings. Two concrete subclasses: `CodeBERT` (`microsoft/codebert-base`) and `CodeGraphBERT` (`microsoft/graphcodebert-base`).

## Selecting it

Task type `embeddings`, settings key `EMBEDDINGS`. `RagAdapter._get_emb_model()` normalizes the configured model id (`.lower().replace("_", "-")`); any value containing `"bert"` routes here — exactly `"codebert"` picks `CodeBERT`, anything else containing `"bert"` picks `CodeGraphBERT`.

## Request payload

```json
{"input": "def add(a, b): return a + b"}
```

`run()` is **synchronous**, returns a flat `List[float]` (mean-pooled over `last_hidden_state` using the attention mask, then L2-normalized).

## VRAM

`get_required_vram()`: `0` on CPU, `800` MB on GPU.

!!! bug "Known issues in this file"
    - `BertEmbedder.__init__` never calls `super().__init__()`, so `self.meta` is never set despite subclassing `BaseModel` — this class isn't actually driven by `model_meta` fields the way other wrappers are.
    - There is no `unload()` method at all (not even a no-op override), so the underlying `torch` model/tokenizer are never freed — relies entirely on `BaseModel.unload()`'s no-op.
    - `run_passage()`/`run_query()` convenience wrappers call `self.run(input_text)` with a **raw string**, but `run()` expects a dict payload and does `payload.get(...)` — calling either of these would raise `AttributeError`. They appear to be unused dead code rather than a real entry point.
