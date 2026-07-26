---
tags:
  - TTS
  - Voice Cloning
  - CPU
  - GPU
---

# Chatterbox

**File:** `app/models/tts_chatterbox.py` · **Class:** `ChatterboxTTSModel(BaseModel)`

Zero-shot voice-cloning TTS wrapping Resemble AI's [Chatterbox](https://github.com/resemble-ai/chatterbox) (multilingual model), with pitch-shift post-processing and automatic idle GPU→CPU offload.

## Selecting it

Alias prefix `chatterbox` under `TTS` settings.

## Loading

`Chatterbox.from_pretrained("resemble-ai/chatterbox-multilingual")`, best-effort `.half()` (FP16) on CUDA, then `.to(self.device)`. On CPU, `torch.set_num_threads(2)` is set at construction time.

## Idle offload

If running on GPU, an idle timer (`IDLE_TIMEOUT = 120` seconds) automatically moves the model back to CPU after 2 minutes of inactivity, and moves it back to GPU transparently on the next request — trading a bit of latency on the first request after idling for freed VRAM in between.

## VRAM

`get_required_vram()`: `0` on CPU, `1200` MB on GPU.

## Request payload

```json
{
  "input": "Hello, this is a cloned voice.",
  "voice": "my_voice_id",
  "language": "en",
  "stream": false,
  "semitones": 0.0,
  "exaggeration": 0.8,
  "temperature": 0.5
}
```

Resolves the reference voice via the shared `voice_service.get_voice_path(voice_id)` (raises `ValueError` if not registered). Streams natively (one of only two engines that do, with [Kokoro](tts_kokoro.md)) when `"stream": true` — pulling from `self._model.synthesize_stream(...)`'s sync generator via `asyncio.to_thread`. `semitones` (non-zero) applies a `pedalboard` pitch shift, same pattern as Kokoro.
