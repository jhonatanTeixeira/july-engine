---
tags:
  - Vision
  - CPU
  - Broken
---

# Tagger (WD14)

**File:** `app/models/tagger.py` · **Class:** `TaggerModel`

Wraps an ONNX WD14-style anime/booru image tagger (default repo `SmilingWolf/wd-swinv2-tagger-v3`, downloaded on-the-fly from the HF Hub), producing weighted danbooru-style tags above a confidence threshold.

## Selecting it

Set `alias` to `tagger` under `VISION` settings — but see the warning below; the current wiring is broken. Instantiated with **no constructor arguments** (`TaggerModel()`) — it doesn't subclass `BaseModel` and has no `backend` parameter at all, so the adapter's configured backend is discarded for this engine.

!!! danger "Broken as currently wired"
    `TaggerModel` has **no `run()` method whatsoever** — its public inference method is `tag(image, custom_threshold=None)`. But `VisionAdapter._analyze` calls every vision engine uniformly as `model.run({"image": img})`, which raises `AttributeError: 'TaggerModel' object has no attribute 'run'` for this engine. The "tagger" vision engine path currently cannot succeed on any request.

## Loading

Downloads `model.onnx` + `selected_tags.csv` via `hf_hub_download`, builds an `onnxruntime.InferenceSession` — **hardcoded to `CPUExecutionProvider`**, no GPU code path exists in this file. Auto-detects the model's expected input resolution (`target_size`, typically 448) from the ONNX session's input shape.

## Inference (once the calling bug above is fixed)

`tag(image, custom_threshold=None)` pads the image to a square canvas (preserving aspect ratio), resizes via bicubic interpolation, converts RGB→BGR, runs the ONNX session, filters by threshold (default `0.35`), strips `rating:`-prefixed tags, and returns:

```json
{"tags": {"1girl": 0.98, "outdoors": 0.87}, "prompt_string": "1girl, outdoors"}
```

## VRAM

`get_required_vram()`: always `0` (ONNX, CPU-only).

## Fixing it

The straightforward fix is on the caller side: `VisionAdapter._analyze` should call `model.tag(img)` instead of `model.run({"image": img})` for the `tagger` engine specifically.
