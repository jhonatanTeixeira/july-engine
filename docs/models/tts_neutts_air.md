---
tags:
  - TTS
  - Voice Cloning
  - CPU
  - GPU
---

# NeuTTS Air

**File:** `app/models/tts_neutts_air.py` · **Class:** `NeuTTSAirModel(BaseModel)`

[Neuphonic NeuTTS Air](https://github.com/neuphonic/neutts-air) — a ~0.7B Qwen2-based on-device TTS model with instant voice cloning, designed CPU-first (it also runs on GPU via `backbone_device`/`codec_device`).

## Selecting it

Alias prefix `neutts` under `TTS` settings maps to engine tag `neutts_air` in `TTSAdapter`'s `_ALIAS_ENGINE_MAP`.

## Loading

Two separate HF repos: `backbone_repo` (default `neuphonic/neutts-air`, overridable via `model_meta["backbone_repo"]` or `["model"]`) and `codec_repo` (default `neuphonic/neucodec`). Both load onto the same device (`cuda` if `backend="gpu"`, else `cpu`) via the `neutts.NeuTTS` constructor.

## VRAM

`get_required_vram()`: `0` on CPU (the intended default backend), `1500` MB on GPU.

## Request payload

```json
{"input": "Hello, this is a cloned voice.", "voice": "my_voice_id", "ref_text": "optional exact transcript of the reference clip"}
```

Returns raw WAV `bytes` (24kHz, via `soundfile`).

## Reference audio transcript — auto-transcription

NeuTTS Air's cloning API needs the **transcript** of the reference audio, not just the audio itself. If `ref_text` isn't supplied in the payload, this model calls the engine's own STT pipeline (`bridge.process_stt`) to transcribe the voice's reference clip automatically, and caches the result per `voice_id` (`self._ref_text_cache`) so transcription only ever runs once per voice for the lifetime of the loaded model. Reference audio *encodings* (`self._model.encode_reference(voice_path)`) are cached the same way, per `voice_id`, in `self._ref_codes_cache`.

This means the very first synthesis request for a given voice is slower (it pays for both STT and reference encoding); subsequent requests for the same voice reuse both caches.
