---
tags:
  - TTS
  - Voice Cloning
  - CPU
  - GPU
---

# Qwen3-TTS

**File:** `app/models/tts_qwen3.py` · **Class:** `FasterQwen3TTSModel(BaseModel)`

Voice-cloning TTS wrapping [`faster_qwen3_tts`](https://github.com/) (a fast inference wrapper for Alibaba's Qwen3-TTS models).

!!! warning "Dependency conflicts with faster-qwen-tts"
    `faster-qwen3-tts` pins `transformers>=4.57,<5` in its own `pyproject.toml`, which can conflict with other packages in this project pinned to `transformers==5.0.0`. If you hit a resolver conflict installing this alongside the rest of `requirements.txt`, consider isolating it (separate venv / sidecar process) rather than relaxing the project's own `transformers` pin.

## Selecting it

Alias prefix `qwen3-tts` or `qwen3_tts` under `TTS` settings (engine tag `qwen3`). Also handled as its own branch in `TTSAdapter._dispatch_stream` (distinct from the Kokoro/Chatterbox native-streaming branch) since its `run()` returns an async-generator object directly under `stream=True` rather than needing a second `await`.

## Loading

Unlike [Faster-Whisper](faster_whisper.md), the model **size/variant is read from `model_meta`**: `model_meta.get("model_size", "12Hz-0.6B-Base")`, used to build the HF repo id `f"Qwen/Qwen3-TTS-{model_size}"`. Loads in `bfloat16` if supported, else `float16`.

## VRAM

`get_required_vram()`: `0` on CPU; on GPU, `3000` MB if `"0.6B"` is in the model size string, else `6500` MB.

## Request payload

```json
{"input": "Hello, this is a cloned voice.", "voice": "my_voice_id", "language": "en", "temperature": 0.7, "stream": false}
```

Resolves voice via `voice_service.get_voice_path`; falls back to the voice's own registered language if `language` isn't given. Text is sanitized (quotes/dashes stripped, a trailing `.` force-appended if missing). Internally always passes `xvec_only=True` to `generate_voice_clone(_streaming)` — not currently exposed via the payload.

## Cleanup

`unload()` is the only TTS wrapper in this engine that delegates to the shared `resource_manager.clear_memory()` rather than doing manual `gc`/`torch.cuda` cleanup directly — worth knowing if you're auditing cleanup behavior across the TTS family, though functionally equivalent.
