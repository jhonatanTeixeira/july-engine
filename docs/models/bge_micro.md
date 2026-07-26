---
tags:
  - Embeddings
  - CPU
  - GPU
---

# BGE Micro

**File:** `app/models/bge_micro.py` · **Class:** `BgeMicroModel(BaseModel)`

Wraps an ONNX export of `BAAI/bge-small-en-v1.5` via `onnxruntime` for CLS-token sentence embeddings. This is the **default/fallback embedder** for the RAG pipeline — `RagAdapter._get_emb_model()` instantiates it whenever the configured embeddings model isn't `multilingual-e5` and doesn't contain `"bert"`.

## Selecting it

Task type `embeddings`, settings key `EMBEDDINGS`. It's the implicit default — no specific alias string is required to reach it.

## Loading

Downloads `onnx/model.onnx` from the HF hub repo via `hf_hub_download`, builds an `onnxruntime.InferenceSession`. On `backend="gpu"`, tries `["CUDAExecutionProvider", "CPUExecutionProvider"]`; if session creation fails on GPU, retries with CPU-only providers.

## Request payload

```json
{"input": "some text to embed"}
```

`run()` is synchronous, returns a flat `List[float]` (CLS token, L2-normalized).

## VRAM

`get_required_vram()`: `100` MB on GPU ("small cost for ONNX on GPU"), `0` on CPU (the default backend).

!!! bug "Known issue"
    In the GPU-load exception handler, the fallback `ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])` references `model_path`, which is only assigned earlier in the `try` block — if `hf_hub_download` itself is what failed (rather than `InferenceSession`), `model_path` would be undefined and the except branch would raise `NameError` instead of falling back cleanly.
