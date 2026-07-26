---
tags:
  - TTS
  - CPU
---

# Piper

**File:** `app/models/tts_piper.py` · **Class:** `PiperModel(BaseModel)`

Fast, fixed-voice TTS via [rhasspy Piper](https://github.com/rhasspy/piper), run as a **subprocess** (`python -m piper`) rather than an in-process model — the only TTS wrapper in this engine that works this way.

## Selecting it

Alias prefix `piper` under `TTS` settings.

## "Loading"

`load()` and `unload()` are both no-ops, and `is_loaded()` **always returns `True`** — there's no persistent in-memory state to track; every `run()` call spins up a fresh subprocess. This means the resource-manager layer will always treat this model as "resident," even though it holds nothing in memory between calls.

## VRAM

`get_required_vram()`: hardcoded `0`, always — CPU-only, no CUDA path exists in this file at all.

## Request payload

```json
{"input": "Hello there!", "voice": "default"}
```

`run()` is **synchronous**. Resolves the voice's `.onnx`/`.onnx.json` files — either from an explicit `hf_path`, or by looking up the registered voice's `piper_path` in the persistence layer's uploaded-voices catalog, downloading from the `rhasspy/piper-voices` HF repo on first use if not already cached. Pipes the text to `python -m piper --model ... --config ... --output_file ...` via stdin, raises `RuntimeError` on non-zero subprocess exit, then reads and returns the resulting WAV file's bytes (deleting the temp file afterward).

No native streaming — `TTSAdapter`'s streaming path falls back to sentence-chunking for this engine, same as [XTTS2](tts_xtts2.md).
