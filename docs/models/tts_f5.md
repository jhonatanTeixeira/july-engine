---
tags:
  - TTS
  - Voice Cloning
  - GPU
---

# F5-TTS

**File:** `app/models/tts_f5.py` · **Class:** `F5TTSModel(BaseModel)`

[F5-TTS (SWivid)](https://github.com/SWivid/F5-TTS) — a ~336M parameter DiT flow-matching TTS model, fast non-autoregressive zero-shot voice cloning from a short reference clip.

## Selecting it

Alias prefix `f5-tts` or `f5_tts` under `TTS` settings maps to engine tag `f5tts` in `TTSAdapter`'s `_ALIAS_ENGINE_MAP`.

## Loading

Loads via `f5_tts.api.F5TTS(model=self.model_variant, device=self.device)`. `model_variant` defaults to `F5TTS_v1_Base`, overridable via `model_meta["variant"]` or `["model"]`.

## VRAM

`get_required_vram()`: `0` on CPU, `3000` MB on GPU (default `backend="gpu"`).

## Request payload

```json
{
  "input": "Hello, this is a cloned voice.",
  "voice": "my_voice_id",
  "ref_text": "optional exact transcript of the reference clip",
  "speed": 1.0,
  "nfe_step": 32,
  "cfg_strength": 2.0,
  "seed": 42
}
```

Returns raw WAV bytes at whatever sample rate `F5TTS.infer()` reports back (not hardcoded, unlike some of the engine's other TTS wrappers).

- `nfe_step` — number of flow-matching function evaluations (quality/speed tradeoff, higher = slower + better).
- `cfg_strength` — classifier-free guidance strength for how closely the output follows the reference voice/style.

## Reference audio transcript — auto-transcription

Same pattern as [NeuTTS Air](tts_neutts_air.md): if `ref_text` isn't supplied, the reference clip is auto-transcribed via `bridge.process_stt` on first use and cached per `voice_id` in `self._ref_text_cache` for the lifetime of the loaded model instance.
