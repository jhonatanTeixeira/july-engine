---
tags:
  - Video
  - GPU
  - SDNQ
---

# Wan2.2 Image-to-Video

**File:** `app/models/wan2_i2v.py` · **Class:** `WanI2VModel(SDNQDiffusionModel)`

Animates a single reference image into a short video clip, using Alibaba's Wan2.2 I2V (14B) in SDNQ-quantized form.

## Selecting it

Set `model` to `wan-i2v`, `wan2-i2v`, or `wan_i2v` under `VIDEO_GENERATION` settings — all three map to the `wan_i2v` engine tag, task type `video_generation`.

## Loading

Always SDNQ mode — no native-diffusers fallback like [Wan2.2 T2V](wan2_t2v.md) has. Default model ID: `Disty0/Wan2.2-I2V-A14B-SDNQ-uint4-svd-r32`. Loads `WanImageToVideoPipeline` in bfloat16 and applies SDNQ's quantized matmul to the transformer (and text encoder, best-effort) when Triton is available.

## VRAM tiers (`WAN_I2V_OFFLOAD` env var)

| Offload | VRAM |
|---|---|
| `sequential` | 2000 MB |
| `cpu` | 4000 MB |
| `none` | 10000 MB |

## Request payload

```json
{
  "image": "<base64 or data: URI>",
  "prompt": "the eagle flaps its wings and takes flight",
  "negative_prompt": "blurry, low quality",
  "height": 480, "width": 832,
  "num_frames": 81, "num_inference_steps": 40,
  "guidance_scale": 3.5, "fps": 16, "seed": -1
}
```

`image` is required — `run()` raises `ValueError` if missing. Accepts a `PIL.Image`, a raw/base64 string (with or without a `data:` prefix), or raw bytes.

## Output

Same streaming contract as [Wan2.2 T2V](wan2_t2v.md): `run()` is an async generator — blocking render in a worker thread, MP4 export, then chunked file streaming via the shared `_stream_file()` helper from [SDNQ Diffusion Base](sdnq_diffusion_base.md).
