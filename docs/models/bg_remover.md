---
tags:
  - Image
  - CPU
  - GPU
---

# Background Removal (rembg)

**File:** `app/models/bg_remover.py` · **Class:** `BgRemoverModel(BaseModel)`

Wraps the [`rembg`](https://github.com/danielgatis/rembg) library's `u2net` ONNX segmentation model to strip image backgrounds.

## Selecting it

Set `alias` to `rembg` under `RESIZE`/image settings — task type `image_remove_background`.

## Loading

Builds an `onnxruntime` session via `rembg.new_session(model_name="u2net", providers=[...])`. On `backend="gpu"`, tries (in order) `CUDAExecutionProvider`, `VulkanExecutionProvider`, `ROCMExecutionProvider` — whichever is actually available per `ort.get_available_providers()` — always appending `CPUExecutionProvider` as a guaranteed fallback.

## Request payload

```json
{"image": "<base64 or data: URI, or a PIL.Image directly>"}
```

`run()` is synchronous, returns a base64-encoded RGBA PNG string (transparent background).

## VRAM

`get_required_vram()`: `200` MB on GPU, `0` on CPU.
