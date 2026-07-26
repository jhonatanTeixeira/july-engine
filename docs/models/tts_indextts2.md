---
tags:
  - TTS
  - Voice Cloning
  - GPU
---

# IndexTTS-2

**File:** `app/models/tts_indextts2.py` · **Class:** `IndexTTS2Model(BaseModel)`

[IndexTTS-2 (IndexTeam)](https://github.com/index-tts/index-tts) — industrial-grade zero-shot voice-cloning TTS with **explicit emotion control**, either from a reference audio clip, a numeric emotion vector, or an alpha-blended emotion reference. The heaviest TTS backend in this engine.

!!! warning "4GB-class GPUs need not apply"
    `get_required_vram()` returns a hardcoded `8000` MB on GPU — deliberately honest, so the orchestrator's VRAM gate rejects this model cleanly (`MemoryError`) on small cards instead of letting it OOM mid-load. IndexTTS 1.5 already needed ~8GB, and v2 adds an embedded emotion model on top.

## Selecting it

Alias prefix `indextts` under `TTS` settings maps to engine tag `indextts2` in `TTSAdapter`'s `_ALIAS_ENGINE_MAP`.

!!! note "Not installed by default"
    The `indextts` pip package is **intentionally not** in `requirements.txt` — `index-tts==2.0.0` requires `torch==2.8.*`, which conflicts with this project's `torch==2.11.0` pin. Install it separately in an isolated venv if you want to use this model: `pip install "indextts @ git+https://github.com/index-tts/index-tts.git"`. This wrapper file is present but unused until that's done.

## Loading

Checkpoint isn't a plain HF Hub download — `_ensure_checkpoint()` uses `huggingface_hub.snapshot_download(repo_id="IndexTeam/IndexTTS-2", local_dir=<checkpoint_dir>)` the first time (checked via presence of `config.yaml`), then loads via `indextts.infer_v2.IndexTTS2(cfg_path=..., model_dir=..., use_fp16=<gpu>, use_cuda_kernel=False, use_deepspeed=False)`.

## Request payload

```json
{
  "input": "Hello, this is a cloned voice with emotion.",
  "voice": "my_voice_id",
  "emotion_voice": "optional_second_voice_id_for_emotion_reference",
  "emo_alpha": 0.9,
  "emo_vector": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "use_random": false
}
```

Returns raw WAV bytes.

Three emotion modes, in priority order:

1. **`emotion_voice`** set → resolves a second registered voice's audio as the emotion reference (`emo_audio_prompt`), blended at `emo_alpha` (default `0.9`).
2. **`emo_vector`** set (and no `emotion_voice`) → passed straight through as IndexTTS2's numeric emotion vector, plus `use_random`.
3. Neither set → plain voice cloning with no explicit emotion conditioning.
