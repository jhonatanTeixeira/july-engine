---
tags:
  - Image
  - GPU
  - SDNQ
---

# Qwen-Image-Edit (SDNQ)

**File:** `app/models/qwen_image_edit.py` · **Class:** `QwenImageEditModel(SDNQDiffusionModel)`

Instruction-guided image editing via Alibaba's Qwen-Image-Edit-2511, SDNQ-quantized. Model: `Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32`.

## Selecting it

Set `model` to `qwen-edit` or `qwen_edit` under `IMAGE_EDIT` settings — `ImageAdapter._detect_engine()` maps both to the `qwen_edit` engine tag.

## Loading

Loads `diffusers.QwenImageEditPlusPipeline` in bfloat16, then applies SDNQ's quantized matmul to the transformer (and text encoder, best-effort) when Triton is available — the same pattern as every other SDNQ pipeline in this engine (see [SDNQ Diffusion Base](sdnq_diffusion_base.md)).

## VRAM tiers (`QWEN_EDIT_OFFLOAD` env var)

| Offload | VRAM |
|---|---|
| `sequential` | 1500 MB |
| `cpu` | 2500 MB |
| `none` | 6000 MB |

## Request payload

```json
{
  "image": "<base64 or data: URI, required>",
  "prompt": "make the sky purple",
  "negative_prompt": "",
  "num_inference_steps": 40,
  "guidance_scale": 1.0,
  "true_cfg_scale": 4.0,
  "num_images_per_prompt": 1,
  "seed": -1
}
```

`image` is required — `run()` raises `ValueError` if missing. `run()` is **synchronous** and returns a single base64-encoded PNG string.

`true_cfg_scale` is a Qwen-Image-Edit-specific parameter (distinct from `guidance_scale`) controlling how strongly the edit instruction is followed versus preserving the source image.
