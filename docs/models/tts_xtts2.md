---
tags:
  - TTS
  - Voice Cloning
  - CPU
  - GPU
---

# XTTS2

**File:** `app/models/tts_xtts2.py` · **Class:** `XTTS2Model(BaseModel)`

Coqui XTTS v2 voice-cloning TTS, wrapping the high-level `TTS.api.TTS` class from [coqui-tts](https://github.com/idiap/coqui-ai-TTS) (hardcoded model name `tts_models/multilingual/multi-dataset/xtts_v2`).

!!! warning "Check the model license before commercial use"
    XTTS v2's weights are distributed under the Coqui Public Model License (CPML), which is non-commercial — see [Licensing](../licensing.md).

## Selecting it

Alias prefix `xtts` under `TTS` settings.

## Loading & idle offload

Same idle-offload concept as [Chatterbox](tts_chatterbox.md) (`IDLE_TIMEOUT = 120`), but implemented with a **`threading.Timer`** instead of an `asyncio` task — a real API-style divergence between the two, though harmless here since `run()` on this class is synchronous.

## VRAM

`get_required_vram()`: `0` on CPU, `2500` MB on GPU.

## Request payload

```json
{"input": "Hello, this is a cloned voice.", "voice": "my_voice_id", "language": "en", "temperature": 0.7}
```

`run()` is **synchronous**, always returns full WAV `bytes` at a fixed 24kHz — **no native streaming**; `TTSAdapter`'s streaming path falls back to sentence-chunking (splitting text into sentences and calling `run()` once per sentence) for this engine.

Text sanitization strips quotes, replaces `-` with a space, and **replaces every `.` with a newline** before synthesis — different from the other TTS wrappers, and worth knowing if input text has many abbreviations/decimals, since each `.` becomes an implicit sentence break for XTTS2's own internal chunking.
