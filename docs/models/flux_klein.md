---
tags:
  - Image
  - GPU
  - SDNQ
---

# FLUX.2 Klein (SDNQ)

**File:** `app/models/flux_klein.py` · **Class:** `FluxKleinPipeline(SDNQDiffusionModel)`

Text-to-image **and** image-to-image (editing) in a single class, sharing one set of SDNQ-quantized weights — the img2img pipeline is cloned from the text2img one at zero extra VRAM cost. Model: `Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic` (4-bit dynamic quantization).

## Selecting it

Set `model` to `flux-klein` or `flux_klein` under `IMAGE_EDIT` / `IMAGE_CREATE` settings — `ImageAdapter._detect_engine()` maps both to the `flux` engine tag.

## Loading

Overrides `PIPELINE_ATTRS = ("model_t2i", "model_i2i")` (see [SDNQ Diffusion Base](sdnq_diffusion_base.md)) since it holds two pipeline objects:

1. `AutoPipelineForText2Image.from_pretrained(...)` in bfloat16 — the SDNQ-quantized weights.
2. SDNQ quantized matmul applied to the transformer (and text encoder, best-effort) if Triton is available.
3. `AutoPipelineForImage2Image.from_pipe(self.model_t2i)` — clones the img2img variant from the already-loaded pipeline's components (no separate weight load).
4. Offload applied to `model_t2i` only (the clone shares the same underlying modules).
5. VAE tiling/slicing enabled unless `ENABLE_VAE=false`.

## VRAM tiers (`FLUX_OFFLOAD` env var)

| Offload | VRAM |
|---|---|
| `sequential` | 1000 MB |
| `cpu` | 1500 MB |
| `none` | 3500 MB |

## Request payload

Text-to-image (no `image` field):

```json
{"prompt": "a cyberpunk knight in neon armor", "width": 512, "height": 512, "num_inference_steps": 4, "guidance_scale": 1.0, "seed": 1337}
```

Image-to-image / editing (`image` present — base64 or data URI):

```json
{"prompt": "same knight, now in a forest at night", "image": "<base64>", "width": 512, "height": 512, "num_inference_steps": 4, "seed": 42}
```

`run()` is **synchronous** (not an async generator — this returns a single image, unlike the video models). Width/height are rounded down to a multiple of **16** — not 8, the pipeline itself rejects anything else with "height and width have to be divisible by 16" — for both text2img and img2img. For img2img, the input image is then resized to match if its size differs.

Returns a base64-encoded PNG string directly.

## NSFW LoRA easter egg

If the request carries header `x-nsfw: 1`, the model lazily loads a LoRA from `models/ExcellentFullNude_F2K4B_1.safetensors` (repo-root-relative) via `load_lora_weights()`, and unloads it again on the next request that doesn't set the header. If the file isn't present, this is a no-op with a warning log — no error.

## Step count override

The `x-max-steps` header takes priority over `num_inference_steps` in the payload, which itself falls back to `4` — the step count this SDNQ 4-bit build is tuned for. Raise it via the header to trade latency for quality without changing the request body, applies to both text-to-image and image-to-image.
