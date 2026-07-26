---
tags:
  - Video
  - CPU
  - GPU
  - Broken
---

# Stable Diffusion Video (legacy)

**File:** `app/models/stable_diffusion_video.py` · **Class:** `LCMVideoPipeline`

An older text-to-video pipeline using Diffusers' `AnimateDiffPipeline` + `MotionAdapter` (`wangfuyun/AnimateLCM`) with an LCM scheduler, loading a single-file SD1.5-based checkpoint. This predates the SDNQ-based video family — [Wan2.2 T2V](wan2_t2v.md), [Wan2.2 I2V](wan2_i2v.md), [LTX-2](ltx2_video.md) — which superseded it and are the ones that actually work today.

!!! danger "Broken as currently wired — dead route via ImageAdapter"
    `ImageAdapter._get_strategy()` constructs this as `LCMVideoPipeline(device=..., use_sequential_offload=True)`, but `LCMVideoPipeline.__init__(self, base_model_path=..., device="cuda", dtype=None)` has **no `use_sequential_offload` parameter and no `**kwargs`** to swallow it — construction raises `TypeError`, caught by a broad `try/except` that leaves `self._strategy = None`, so the `video` alias route always ends in `ValueError("no local video-generation model available...")`.

    Even past that, `ImageAdapter._generate_video()` calls `strategy.run(payload)` expecting an async-generator — but this class has no `run()` method at all; its public entry point is `generate_video(prompt, negative_prompt="", num_frames=16, width=384, height=384, steps=6, guidance_scale=1.5, seed=42)`, which is synchronous and returns a plain `list` of PIL frames, not a streamed file.

    Like `Pix2Pix`, this class doesn't subclass `BaseModel` and doesn't take `backend`/`model_meta`.

## Use [Wan2.2](wan2_t2v.md) or [LTX-2](ltx2_video.md) instead

For any new video-generation work, prefer the SDNQ video family, which is wired correctly end-to-end and streams output rather than buffering a full clip in memory:

- [Wan2.2 Text-to-Video](wan2_t2v.md)
- [Wan2.2 Image-to-Video](wan2_i2v.md)
- [LTX-2](ltx2_video.md) (also produces synchronized audio)

## What it does today (in isolation, if called directly)

`load()` builds the `AnimateDiffPipeline` and unconditionally enables VAE slicing/tiling and `enable_model_cpu_offload()` (no flags to toggle this, unlike the sibling [LCM FaceID](stable_diffusion_lcm.md) class). `generate_video(...)` runs the pipeline and returns a list of frames — has a `if __name__ == "__main__":` demo block using `diffusers.utils.export_to_gif`. `get_required_vram()`: `0` on CPU, else a hardcoded `4000` MB (with an unreachable `return 3800` dead statement immediately after it).
