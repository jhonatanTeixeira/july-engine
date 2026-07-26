---
tags:
  - Vision
  - CPU
  - Broken
---

# Emotion Detection

**File:** `app/models/emotion.py` · **Class:** `EmotionModel`

Wraps a fixed ONNX FER+ facial-emotion classifier (`emotion-ferplus-8.onnx`, expected at `storage/models/emotion-ferplus-8.onnx` — **not** downloaded from the HF Hub) run against a face crop produced by an injected `FaceDetector`. Classifies into 8 classes: `neutral`, `happiness`, `surprise`, `sadness`, `anger`, `disgust`, `fear`, `contempt`.

## Selecting it

Set `alias` to `emotion` under `VISION` settings — but see the warning below; the current wiring is broken.

!!! danger "Broken as currently wired"
    `EmotionModel.run()` expects a **raw `PIL.Image` (or list of them)** as its first positional argument — not a payload dict, and it doesn't subclass `BaseModel`. But `VisionAdapter._analyze` calls it as `model.run({"image": img})`, passing a **dict**. Since a dict isn't a list, `run()` falls through to `_run_single({"image": img})`, which immediately calls `.convert('RGB')` on that dict — raising `AttributeError: 'dict' object has no attribute 'convert'`. The "emotion" vision engine path currently cannot succeed on any request.

## Loading

`load()` builds an `onnxruntime.InferenceSession` with `providers=['CPUExecutionProvider']` — **hardcoded to CPU regardless of `backend`**. There is no `unload()` method at all.

## Inference (once the calling bug above is fixed)

`run(image)` (or a list of images) crops to the first detected face via the injected `face_detector.detect_faces(...)` (falling back to the whole image if no detector/no face found), converts to grayscale, resizes to 64×64, and runs the ONNX session — returning the dominant emotion label as a string (or `"Empty face crop"` if the crop was empty).

## VRAM

`get_required_vram()`: always `0` (ONNX, CPU-only).

## Fixing it

The straightforward fix is on the caller side: `VisionAdapter._analyze` should call `model.run(img)` (the decoded `PIL.Image` directly) instead of `model.run({"image": img})` for the `emotion` engine specifically.
