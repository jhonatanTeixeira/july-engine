---
tags:
  - Overview
---

# Models Overview

Every entry below is a concrete model wrapper in `app/models/`, one `BaseModel` subclass (or, for a few pre-existing files, a set of related classes) per file. Adapters in `app/adapters/` resolve which wrapper to instantiate based on the `model`/`alias` field of a settings entry — see [Architecture → Request Flow](../architecture/request-flow.md) for how a request reaches a model, and [Known Issues](../known-issues.md) for a consolidated list of the broken/dead wrappers flagged below.

## Chat / LLM

| Model | File | Backend | Notes |
|---|---|---|---|
| [GGUF (llama.cpp) Adapter](gguf_adapter.md) | `gguf_adapter.py` | CPU/GPU | Wraps the vendored `llama_gguf.GGUF` class — the engine's only chat/text backend today. |

## Vision (VLM & Image Understanding)

| Model | File | Backend | Notes |
|---|---|---|---|
| [FastVLM](fastvlm.md) | `fastvlm.py` | CPU/GPU | Apple FastVLM-0.5B, 4-bit on GPU. |
| [Moondream](moondream.md) | `moondream.py` | — | :material-alert: **Non-functional** — hard-coded `raise` and a constructor mismatch make this engine unusable as wired. |
| [Molmo](molmo.md) | `molmo.py` | CPU/GPU | SDNQ-quantized Molmo-7B-O, `transformers`-based. |
| [Emotion Detection](emotion.md) | `emotion.py` | CPU only | :material-alert: **Broken** — `VisionAdapter` calls it with the wrong argument shape. |
| [Tagger (WD14)](tagger.md) | `tagger.py` | CPU only | :material-alert: **Broken** — `VisionAdapter` calls a `run()` method this class doesn't have. |

## Embeddings

| Model | File | Backend | Notes |
|---|---|---|---|
| [BERT (CodeBERT/GraphCodeBERT)](bert.md) | `bert.py` | CPU/GPU | Two aliases sharing one base class. |
| [BGE Micro](bge_micro.md) | `bge_micro.py` | CPU/GPU | ONNX `bge-small-en-v1.5` — the RAG default embedder. |
| [Multilingual E5](multilingual_e5.md) | `multilingual_e5.py` | CPU/GPU | :material-alert-outline: query/passage prefixing has a known bug (see page). |

## Entity Extraction

| Model | File | Backend | Notes |
|---|---|---|---|
| [GLiNER2](gliner2_extractor.md) | `gliner2_extractor.py` | CPU/GPU | Zero-shot structured entity extraction. |

## Text-to-Speech

| Model | File | Backend | Voice cloning? |
|---|---|---|---|
| [Kokoro](tts_kokoro.md) | `tts_kokoro.py` | CPU/GPU | No — fixed voice catalog. |
| [Chatterbox](tts_chatterbox.md) | `tts_chatterbox.py` | CPU/GPU | Yes. |
| [XTTS2](tts_xtts2.md) | `tts_xtts2.py` | CPU/GPU | Yes. |
| [Piper](tts_piper.md) | `tts_piper.py` | CPU | No — fixed ONNX voices. |
| [Qwen3-TTS](tts_qwen3.md) | `tts_qwen3.py` | CPU/GPU | Yes. |
| [NeuTTS Air](tts_neutts_air.md) | `tts_neutts_air.py` | CPU-first | Yes, instant cloning. |
| [F5-TTS](tts_f5.md) | `tts_f5.py` | GPU | Yes, zero-shot. |
| [IndexTTS-2](tts_indextts2.md) | `tts_indextts2.py` | GPU (~8GB) | Yes, plus explicit emotion control. |

## Speech-to-Text

| Model | File | Backend | Notes |
|---|---|---|---|
| [Faster-Whisper](faster_whisper.md) | `faster_whisper.py` | CPU/GPU | CTranslate2 Whisper. |

## Image Editing & Generation

| Model | File | Backend | Notes |
|---|---|---|---|
| [InstructPix2Pix](pix2pix.md) | `pix2pix.py` | CPU/GPU | :material-alert: **Broken** — `ImageAdapter` imports a class name (`Pix2PixPipeline`) that doesn't exist in this file. |
| [Stable Diffusion LCM / FaceID](stable_diffusion_lcm.md) | `stable_diffusion_lcm.py` | CPU/GPU | Fast LCM generation + IP-Adapter FaceID — works correctly via the `lcm` alias. |
| [FLUX.2 Klein (SDNQ)](flux_klein.md) | `flux_klein.py` | GPU | Text2img + img2img sharing one set of weights. |
| [Qwen-Image-Edit (SDNQ)](qwen_image_edit.md) | `qwen_image_edit.py` | GPU | Instruction-guided image editing. |

## Image Utilities

| Model | File | Backend | Notes |
|---|---|---|---|
| [Background Removal (rembg)](bg_remover.md) | `bg_remover.py` | CPU/GPU | u2net via `rembg`. |
| [Resizers & Upscalers](image_resizer.md) | `image_resizer.py` | CPU/GPU | 8 strategies in one file — Pillow, OpenCV, GFPGAN, CodeFormer, Real-ESRGAN, etc. |

## Video Generation

| Model | File | Backend | Notes |
|---|---|---|---|
| [Wan2.2 Text-to-Video](wan2_t2v.md) | `wan2_t2v.py` | GPU | SDNQ or native diffusers, streamed MP4 output. |
| [Wan2.2 Image-to-Video](wan2_i2v.md) | `wan2_i2v.py` | GPU | Streamed MP4 output. |
| [LTX-2](ltx2_video.md) | `ltx2_video.py` | GPU | Video **+ audio**, muxed via ffmpeg. |
| [Stable Diffusion Video (legacy)](stable_diffusion_video.md) | `stable_diffusion_video.py` | CPU/GPU | Predates the SDNQ video family. |

## 3D Generation

| Model | File | Backend | Notes |
|---|---|---|---|
| [Trellis2 (stub)](trellis2.md) | `trellis2.py` | — | Deliberately unimplemented — `run()` raises `NotImplementedError`. |

## Shared Infrastructure

| Component | File | Notes |
|---|---|---|
| [SDNQ Diffusion Base](sdnq_diffusion_base.md) | `sdnq_diffusion_base.py` | Shared lifecycle base class for the SDNQ-quantized diffusers pipelines above. |
