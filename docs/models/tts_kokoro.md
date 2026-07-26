---
tags:
  - TTS
  - CPU
  - GPU
---

# Kokoro

**File:** `app/models/tts_kokoro.py` · **Class:** `KokoroTTSModel(BaseModel)`

Wraps the [`kokoro`](https://github.com/hexgrad/kokoro) library's `KPipeline` — a lightweight, fast, per-language TTS pipeline. This is the engine's **implicit default TTS engine**: `TTSAdapter._load_tts_config()` defaults `voice` to `af_heart` (a Kokoro voice ID) and `language` to `"a"` when nothing else is configured.

## Selecting it

Alias prefix `kokoro` under `TTS` settings.

## Loading

`load()` is deliberately a **no-op** — the real `KPipeline` is lazily built inside `run()` via `_ensure_pipeline(lang_code)`, and **reloads whenever the requested language changes** (`if self._pipeline is None or lang_code != self._lang_code: ...`). Switching languages mid-session silently discards and rebuilds the pipeline.

## VRAM

`get_required_vram()`: `0` on CPU, `500` MB on GPU.

## Request payload

```json
{"input": "Hello there!", "voice": "af_heart", "language": "a", "stream": false, "semitones": 0.0}
```

Returns either full WAV `bytes`, or (if `"stream": true`) an async generator of WAV chunks — Kokoro is one of only two engines (with [Chatterbox](tts_chatterbox.md)) that stream natively rather than via sentence-chunking fallback.

`semitones` (optional, non-zero) applies a pitch shift via `pedalboard` if installed — falls back to a no-op with a warning if not.

See the [full built-in voice catalog](../configuration/admin-panel.md) for all 54 available `af_*`/`am_*`/etc. voice IDs, surfaced in the admin panel's Voices tab.
