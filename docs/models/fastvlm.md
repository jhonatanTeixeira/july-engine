---
tags:
  - Vision
  - CPU
  - GPU
---

# FastVLM

**File:** `app/models/fastvlm.py` · **Class:** `FastVLMModel(BaseModel)`

Wraps `transformers.AutoModelForCausalLM` for Apple's `FastVLM-0.5B` — a small, fast vision-language model for image captioning/chat, using a custom `<image>` token protocol (`IMAGE_TOKEN_INDEX = -200`).

## Selecting it

Set `alias` (or `model`) to `fastvlm` under `VISION` settings.

## Loading

On GPU (`torch.cuda.is_available()`): 4-bit NF4 quantization via `BitsAndBytesConfig`, `device_map="auto"`, `torch_dtype=torch.float16`, `attn_implementation="sdpa"`. On CPU: `device_map="cpu"`, `torch_dtype=torch.float32`.

## VRAM

`get_required_vram()`: `0` on CPU, `1200` MB on GPU.

## Request payload

Accepts either OpenAI-style chat messages (extracts the last message's text + image_url parts) or a flat `{"prompt": ..., "image"/"images": ...}` shape. Images may be a `PIL.Image`, base64 (with or without a `data:` prefix), or raw bytes.

```json
{"prompt": "What's in this image?", "image": "<base64>"}
```

Returns a plain string caption, or — when the request used the `messages` shape — an OpenAI-chat-completion-shaped dict.

!!! bug "Known issues"
    - `run()` is defined **synchronously**, and blocks the event loop thread since `VisionAdapter` calls it directly rather than through the (unused) `async def chat()` wrapper that would offload it via `asyncio.to_thread`.
    - Multi-image requests silently only process the first image: the adapter's batch path checks `hasattr(model, "run_batch")`, but this class only exposes a **private** `_run_batch` (not `run_batch`), so the `hasattr` check is always `False` and any images beyond the first are discarded without error.
