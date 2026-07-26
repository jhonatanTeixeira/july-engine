---
tags:
  - Image
  - CPU
  - GPU
---

# Stable Diffusion LCM / FaceID

**File:** `app/models/stable_diffusion_lcm.py` · **Class:** `LCMFaceIDPipeline`

A Latent Consistency Model (LCM) Stable Diffusion 1.5 pipeline combined with Tencent's IP-Adapter FaceID Plus for face-conditioned generation, with optional SDNQ int8 dynamic quantization. Equivalent in spirit to an A1111 LCM + FaceID setup. There is no separate "plain LCM" class in this file — LCM and FaceID are unified in one class, with the `use_face_id` flag controlling whether the FaceID branch is used at inference time.

## Selecting it

Set `alias` to `lcm` under `IMAGE_CREATE`/`IMAGE_EDIT` settings — this is one of the two image-generation routes confirmed to work correctly end-to-end (the other being the SDNQ pipelines); `ImageAdapter._get_strategy()` constructs it as `LCMFaceIDPipeline(device="cuda" if gpu else "cpu", use_sequential_offload=True)`, which matches this class's real constructor signature.

## Loading

Unlike most models in this directory, this class doesn't subclass `BaseModel` and doesn't take `backend`/`model_meta` — it takes `device` directly, plus many optimization flags: `use_xformers`, `use_cpu_offload`, `use_sequential_offload`, `ip_adapter_scale`, `use_face_id` (default `True`), `use_vae_slicing`, `use_vae_tiling`, `use_sdnq`, `sdnq_dtype`.

Default weight paths (repo-root-relative): base model `models/coldfleshRealisticLCM_v10.safetensors`, IP-Adapter `models/ip-adapter-faceid-plus_sd15.bin`, image encoder `models/image_encoder` (falls back to the HF repo `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` if the local dir is missing).

Load sequence: base pipeline (`from_single_file` for local `.safetensors`/`.ckpt`, else `from_pretrained`) → optional SDNQ int8 quantization of the UNet → LCM scheduler → (if `use_face_id`) IP-Adapter FaceID Plus + InsightFace `FaceAnalysis("buffalo_l")` for face embeddings — **InsightFace always runs on CPU** (the CUDA execution-provider line is present but commented out in the source, regardless of `device`) → memory optimizations (xformers, VAE slicing/tiling, one of sequential/model/no CPU offload).

## VRAM

`get_required_vram()`: `0` on CPU; on GPU, `2100` MB with sequential offload, `2800` MB with model-level offload, `3500` MB with no offload.

## Inference — callable, not `run()`

This class has no `run()` method; it's invoked directly via `__call__`:

```python
images = pipeline(
    prompt="a portrait of a person smiling",
    face_image="<base64, optional>",
    negative_prompt="",
    num_inference_steps=10, guidance_scale=1.5,
    width=512, height=512, seed=42,
    ip_adapter_scale=None,   # falls back to the instance default
)
```

Three behaviors depending on `use_face_id` and whether `face_image` is given:

1. `use_face_id=False` → plain LCM text-to-image.
2. `use_face_id=True` + `face_image` provided → detects/crops the face via InsightFace, extracts its embedding, and conditions generation on it via IP-Adapter FaceID Plus.
3. `use_face_id=True` + no `face_image` → a "bypass" mode using a dummy black image and zero-valued embeddings with `s_scale=0.0`, effectively disabling FaceID conditioning without needing a second code path.

`ImageAdapter` converts the first returned `PIL.Image` to a base64 PNG.
