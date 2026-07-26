---
tags:
  - Infrastructure
  - GPU
  - SDNQ
---

# SDNQ Diffusion Base

**File:** `app/models/sdnq_diffusion_base.py` · **Class:** `SDNQDiffusionModel(BaseModel)`

Shared lifecycle base class for every local `diffusers` pipeline loaded from an [SDNQ](https://github.com/Disty0/sdnq)-quantized Hugging Face repo: [Wan2.2 T2V](wan2_t2v.md), [Wan2.2 I2V](wan2_i2v.md), [FLUX.2 Klein](flux_klein.md), [Qwen-Image-Edit](qwen_image_edit.md), and [LTX-2](ltx2_video.md) all subclass it. It is never selected directly by an adapter — it exists purely so those five pipelines don't each reimplement the same offload/VRAM/streaming boilerplate.

## What it provides

| Method | Behavior |
|---|---|
| `is_loaded()` | `True` once the first attribute in `PIPELINE_ATTRS` is set. |
| `unload()` | Clears every attribute in `PIPELINE_ATTRS`, then `gc.collect()` + `torch.cuda.empty_cache()`/`ipc_collect()`. |
| `get_required_vram(payload)` | `0` on CPU backend; otherwise looks up the current `SDNQ_OFFLOAD`-style env var value (`sequential`/`cpu`/`none`) in `VRAM_TIERS`. |
| `_apply_offload(pipeline_obj)` | Applies `enable_sequential_cpu_offload()`, `enable_model_cpu_offload()`, or a plain `.to(self.device)`, based on the same env var. |
| `_stream_file(path)` | Async generator that reads a file in 1MB chunks and deletes it once fully consumed — how video subclasses avoid buffering a whole clip in memory before returning it. |

## Subclass contract

A subclass sets class attributes and implements `load()` + its own `run()`:

```python
class MyPipeline(SDNQDiffusionModel):
    DEFAULT_MODEL_ID = "org/My-Model-SDNQ-uint4"
    OFFLOAD_ENV_VAR = "MY_MODEL_OFFLOAD"          # env var name this model's offload tier reads from
    VRAM_TIERS = {"sequential": 1500, "cpu": 2500, "none": 6000}
    PIPELINE_ATTRS = ("pipeline",)                 # or e.g. ("model_t2i", "model_i2i") for two shared pipelines

    def load(self, n_ctx=None, num_layers=None):
        if self.is_loaded():
            return
        # build self.pipeline, then:
        self._apply_offload(self.pipeline)
```

`PIPELINE_ATTRS` defaults to `("pipeline",)` but [FLUX.2 Klein](flux_klein.md) overrides it to `("model_t2i", "model_i2i")` since its img2img pipeline is cloned from the same weights via `AutoPipelineForImage2Image.from_pipe(...)` — no extra VRAM, just a second pointer that also needs to be cleared on unload.

## Offload tiers

Each subclass's `OFFLOAD_ENV_VAR` controls both the VRAM estimate and the actual `diffusers` offload mode applied at load time:

| Env value | `_apply_offload` behavior | Typical use |
|---|---|---|
| `sequential` (default) | `enable_sequential_cpu_offload()` | Lowest VRAM footprint — layers move to GPU one at a time. 4GB-class cards. |
| `cpu` | `enable_model_cpu_offload()` | Whole submodules move to GPU per call — faster than sequential, more VRAM. |
| `none` | `.to(self.device)` | Full model resident on GPU — fastest, highest VRAM. |

## Streaming rationale

Every video-producing subclass (`wan2_t2v`, `wan2_i2v`, `ltx2_video`) returns an **async generator** from `run()` instead of a full byte blob. The render itself is blocking `diffusers` code, so `run()` offloads it via `asyncio.to_thread(self._render, payload)`, writes the result to a temp file, then hands that file to `_stream_file()` — which is what actually gets returned/iterated by the orchestrator (it already detects `hasattr(result, "__aiter__")` and streams it back to the HTTP client without ever loading the whole clip into RAM).

Every subclass also holds its own `self._inference_lock = asyncio.Lock()` (set in `SDNQDiffusionModel.__init__`) so concurrent requests against the same warm pipeline instance are serialized rather than corrupting each other's generation state.
