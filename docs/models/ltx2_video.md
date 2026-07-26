---
tags:
  - Video
  - Audio
  - GPU
  - SDNQ
---

# LTX-2

**File:** `app/models/ltx2_video.py` · **Class:** `LTX2Model(SDNQDiffusionModel)`

Text-to-video (optionally image-conditioned) generation via Lightricks' LTX-2, the only video model in this engine that also produces **synchronized audio** — muxed into the final MP4 via `ffmpeg`.

## Selecting it

Set `model` to `ltx2`, `ltx-2`, or `ltx_2` under `VIDEO_GENERATION` settings — all map to the `ltx2` engine tag, task type `video_generation`. Default model ID: `Disty0/LTX-2-SDNQ-4bit-dynamic`.

## Loading

Uses the generic `diffusers.DiffusionPipeline.from_pretrained(...)`, which auto-resolves the concrete LTX-2 pipeline class from the repo's `model_index.json` (rather than importing a named pipeline class directly, as the Wan2 models do). Applies SDNQ quantized matmul to the transformer when Triton is available, and enables VAE tiling if the pipeline supports it.

## VRAM tiers (`LTX2_OFFLOAD` env var)

| Offload | VRAM |
|---|---|
| `sequential` | 1500 MB |
| `cpu` | 3000 MB |
| `none` | 7000 MB |

## Request payload

```json
{
  "prompt": "a calm lake at sunrise, birds singing",
  "negative_prompt": "blurry, low quality",
  "image": "<optional base64 reference image>",
  "width": 768, "height": 512,
  "num_frames": 121, "frame_rate": 25,
  "num_inference_steps": 40, "guidance_scale": 4.0,
  "audio_sample_rate": 24000
}
```

`image` is optional — when omitted, generation is pure text-to-video; when present, it's decoded and passed straight through to the pipeline call.

!!! warning "Audio sample rate is a best-effort default"
    At the time this wrapper was written, LTX-2's model card didn't document an exact output audio sample rate. `24000` Hz is used as a reasonable default for diffusers audio pipelines and is overridable via `audio_sample_rate` in the payload — verify against the real pipeline output before relying on it in production.

## Output & audio muxing

`_render()` calls the pipeline, which returns `(video, audio)`. The video frames are exported to a temp MP4 via `export_to_video`; if `audio` is non-`None`, it's written to a temp WAV (`_write_wav`, converting float `[-1, 1]` samples to int16 PCM if needed) and muxed into the video with `ffmpeg -c:v copy -c:a aac -shortest` (via `imageio_ffmpeg.get_ffmpeg_exe()`, so no system-wide ffmpeg install is required). Both intermediate files are deleted once muxing completes.

Like the other video models, `run()` is an async generator that streams the final muxed (or video-only, if the pipeline returned no audio) MP4 in chunks via the shared `_stream_file()` helper — see [SDNQ Diffusion Base](sdnq_diffusion_base.md).
