---
tags:
  - Video
  - GPU
  - SDNQ
---

# Wan2.2 Text-to-Video

**File:** `app/models/wan2_t2v.py` · **Class:** `Wan2T2VPipeline(SDNQDiffusionModel)`

Text-to-video generation via Alibaba's Wan2.2 (14B, MoE-style dual-transformer), streamed back as MP4. Supports two load modes depending on the configured model ID.

## Selecting it

Set `model` (alias) to one of `wan-t2v`, `wan2-t2v`, `wan_t2v` under the `VIDEO_GENERATION` settings key — `ImageAdapter._detect_engine()` maps all three to the same `wan_t2v` engine tag, task type `video_generation`.

## Load modes

`_is_sdnq_model()` checks `meta["variant"] == "sdnq"` (falling back to `"sdnq" in model_id.lower()`) to decide:

- **SDNQ mode** (default model ID `Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32`): loads `WanPipeline` + `AutoencoderKLWan` in bfloat16, then applies SDNQ's quantized matmul to the transformer (and text encoder, best-effort) if Triton is available.
- **Native diffusers mode** (e.g. `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`): loads the same pipeline classes without pre-quantized weights, then stacks on its own optimizations — VAE slicing/tiling, attention slicing, xformers if available, optional SDNQ runtime quantization (`WAN_SDNQ=1`, default on) or `torchao` int8 (`WAN_SDNQ=0`), and `torch.compile(mode="reduce-overhead")` — but **only** when the offload mode isn't `sequential`/`cpu` (CUDA Graphs from `reduce-overhead` are incompatible with accelerate's CPU-offload hooks, which move tensors between devices on every forward pass).

## VRAM tiers

| Offload (`WAN_OFFLOAD`) | SDNQ model | Native diffusers model |
|---|---|---|
| `sequential` | 2000 MB | 2000 MB |
| `cpu` | 4000 MB | 3000 MB |
| `none` | 10000 MB | 5000 MB |

## Request payload

```json
{
  "prompt": "A majestic eagle soaring over snow-capped mountains, cinematic, 4K",
  "negative_prompt": "blurry, low quality, distorted",
  "height": 480, "width": 832,
  "num_frames": 81, "num_inference_steps": 40,
  "guidance_scale": 4.0, "guidance_scale_2": 3.0,
  "fps": 16, "seed": 42
}
```

`guidance_scale_2` is only sent to the pipeline when running in SDNQ mode (the SDNQ checkpoint is CFG-distilled with a second guidance term; the native diffusers pipeline doesn't accept it).

## Output

`run()` is an **async generator** — it renders in a worker thread (`asyncio.to_thread`), exports frames to a temp MP4 via `diffusers.utils.export_to_video`, then streams the file in 1MB chunks (deleting it once fully read). See [SDNQ Diffusion Base](sdnq_diffusion_base.md) for why.

## Standalone testing

The file has a `if __name__ == "__main__":` CLI (`python -m app.models.wan2_t2v --prompt "..." --model ...`) for rendering a clip directly without the FastAPI server, with `--offload`/`--no-compile`/`--no-sdnq` overrides.
