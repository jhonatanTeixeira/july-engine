---
tags:
  - Image
  - CPU
  - GPU
  - Broken
---

# InstructPix2Pix

**File:** `app/models/pix2pix.py` · **Class:** `Pix2Pix`

Instruction-guided image editing wrapping Diffusers' `StableDiffusionInstructPix2PixPipeline` (`timbrooks/instruct-pix2pix`).

!!! danger "Broken as currently wired — dead route via ImageAdapter"
    `ImageAdapter._get_strategy()` tries `from ..models.pix2pix import Pix2PixPipeline` — but this file only defines a class named **`Pix2Pix`**, not `Pix2PixPipeline`. The import raises `ImportError`, caught by a broad `try/except` that logs a warning and leaves `self._strategy = None`, so the `pix2pix` image-edit route always ends in `ValueError("no local image-edit model available...")`.

    Even if the class name were fixed, the adapter calls `Pix2PixPipeline(device=..., use_sequential_offload=True)`, but `Pix2Pix.__init__(self, backend="gpu")` only accepts `backend` — no `device` or `use_sequential_offload` kwarg — so construction would still fail with `TypeError`. And even past that, the adapter calls `strategy.run({...})` with a single payload dict, while `Pix2Pix.run(self, image_data: str, prompt: str, **kwargs)` expects two separate positional arguments.

    `Pix2Pix` also doesn't subclass `BaseModel` and has no `unload()` method.

## What would need to change to make this work

1. Rename the class to `Pix2PixPipeline` (or fix the import to `Pix2Pix`).
2. Accept `device`/`use_sequential_offload` in `__init__` (or drop those kwargs on the caller side).
3. Change `run()` to accept a single payload dict, matching every other model in this directory.
4. Add an `unload()` method.

## What it does today (in isolation, if called directly)

`load()` builds the pipeline in float16 on GPU / float32 on CPU, and swaps in `EulerAncestralDiscreteScheduler`. `run(image_data, prompt, **kwargs)` decodes a base64 image, optionally resizes, runs the pipeline with `num_inference_steps=20, image_guidance_scale=1.5`, and returns a base64 PNG string. `get_required_vram()`: `0` on CPU, `2800` MB on GPU.

For a working local image-editing alternative today, see [Qwen-Image-Edit (SDNQ)](qwen_image_edit.md) or [FLUX.2 Klein](flux_klein.md) (img2img mode), or [Stable Diffusion LCM / FaceID](stable_diffusion_lcm.md) for face-conditioned generation.
