---
tags:
  - Vision
  - GPU
  - SDNQ
---

# Molmo

**File:** `app/models/molmo.py` · **Class:** `MolmoModel(BaseModel)`

Vision-language model for image captioning/description, wrapping AllenAI's Molmo-7B-O in an SDNQ-quantized form: `CalamitousFelicitousness/Molmo-7B-O-0924-SDNQ-UINT4-SVD-R32`.

## Selecting it

Set `alias` (or `model`) to `molmo` under `VISION` settings — `VisionAdapter._get_vision_model()` matches it literally.

## Loading

Unlike the diffusers-based image/video models, Molmo is a plain `transformers.AutoModelForCausalLM` + `AutoProcessor` pair (`trust_remote_code=True`, `torch_dtype="auto"`, `device_map="auto"` on GPU or `"cpu"` otherwise). After loading, it makes a best-effort attempt to apply SDNQ's `apply_sdnq_options_to_model(..., use_quantized_matmul=True)` post-load — there's no established precedent elsewhere in this codebase for combining SDNQ with a plain `transformers` causal LM (only `diffusers` pipelines use this pattern), so failure here is logged as a warning and non-fatal, not raised.

## VRAM

`get_required_vram()` returns a hardcoded `6000` MB on GPU, `0` on CPU.

## Request payload

```json
{"image": "<base64 or data: URI>", "prompt": "Describe this image."}
```

`run()` is **synchronous**, returns a plain `str`. Returns `""` immediately if no `image` is provided. Internally uses `self._processor.process(...)` + `self._model.generate_from_batch(...)` with `max_new_tokens=200` and a `<|endoftext|>` stop string — the standard Molmo inference recipe, not something specific to this engine.
