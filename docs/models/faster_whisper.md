---
tags:
  - STT
  - CPU
  - GPU
---

# Faster-Whisper

**File:** `app/models/faster_whisper.py` · **Class:** `FasterWhisperModel(BaseModel)`

The engine's only STT backend — wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a CTranslate2 reimplementation of OpenAI Whisper), with a noise-reduction pre-processing pass before transcription.

## Selecting it

There's no alias map for STT — `STTAdapter` unconditionally instantiates this class. Task type `stt`, settings key `STT`.

!!! note "Model size comes from an environment variable, not settings"
    `self.model_size = os.environ.get("STT_MODEL", "medium")` — unlike every other model in this engine, the Whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`/`large-v3-turbo`/...) is read from the `STT_MODEL` env var, not from `model_meta`. Per-request or per-alias model-size selection isn't supported.

## Loading

`compute_type = "float16"` on GPU (if CUDA available), `"int8"` on CPU.

## VRAM

Hardcoded per `model_size`: `tiny` 250MB, `base` 350MB, `small` 600MB, `medium` 1500MB, `large`/`large-v1`/`large-v2`/`large-v3` 3000MB, `large-v3-turbo` 2500MB (default fallback: 1500MB). `0` on CPU backend.

## Request payload

```json
{"audio": "<raw audio bytes>", "language": "en"}
```

`run()` is **synchronous** (not `async`, despite `BaseModel.run` being declared `async def` — works because the caller, `STTAdapter.run`, is itself async and just returns the plain string without needing to await this method specially). Decodes audio via `soundfile`, downmixes to mono if stereo, runs `noisereduce.reduce_noise(prop_decrease=0.8)` before transcription, then `WhisperModel.transcribe(..., vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))`. Returns a plain transcript string (segments joined with spaces) — no streaming, no per-segment timestamps exposed.
