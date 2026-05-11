import asyncio
import io
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Union

import numpy as np
import soundfile as sf

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.FasterQwen3TTS")


class FasterQwen3TTSModel(BaseModel):
    """
    FasterQwen3 TTS with two output forms:
      - stream=False → bytes (full WAV)
      - stream=True  → AsyncGenerator[bytes] (WAV chunks)

    The underlying library yields a sync generator; we wrap it as an
    async generator so the orchestrator's __aiter__ detection works
    uniformly across all streaming TTS models.
    """

    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.device = "cpu"
        self._model = None
        self.model_size = (model_meta or {}).get("model_size", "12Hz-0.6B-Base")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu":
            return 0
        return 3000 if "0.6B" in self.model_size else 6500

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        logger.info(f"FasterQwen3TTS: Loading {self.model_size} on {self.device}")
        self._model = FasterQwen3TTS.from_pretrained(
            f"Qwen/Qwen3-TTS-{self.model_size}",
            device=self.device,
            dtype=dtype,
        )
        logger.info("FasterQwen3TTS loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        if self._model:
            del self._model
            self._model = None
        from ..resource_manager import resource_manager
        resource_manager.clear_memory()
        logger.info("FasterQwen3TTS unloaded.")

    async def run(self, payload: Dict[str, Any], **kwargs) -> Union[bytes, AsyncGenerator[bytes, None]]:
        from ..services.voice_service import voice_service

        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "default")
        language = payload.get("language", "en")
        temperature = payload.get("temperature", 0.7)
        stream = payload.get("stream", False)

        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            raise ValueError(f"FasterQwen3TTS: Voice '{voice_id}' not found")
        voice_path, voice_lang = voice_res
        target_lang = language or voice_lang

        if self._model is None:
            self.load()

        text = text.replace('"', "").replace("-", " ").strip()
        if not text.endswith((".", "!", "?")):
            text += "."

        if stream:
            return self._stream_async(text, target_lang, voice_path, temperature)

        return await asyncio.to_thread(self._generate_full, text, target_lang, voice_path, temperature)

    # ------------------------------------------------------------------
    # Streaming form: wraps sync generator as async generator
    # The library yields (wav_chunk, sr, timing) tuples synchronously;
    # we pull each tuple in a thread to avoid blocking the event loop.
    # ------------------------------------------------------------------

    async def _stream_async(self, text, language, voice_path, temperature) -> AsyncGenerator[bytes, None]:
        sync_gen = self._model.generate_voice_clone_streaming(
            text=text, language=language, ref_audio=voice_path,
            temperature=temperature, xvec_only=True,
        )

        def next_item():
            try:
                return next(sync_gen)
            except StopIteration:
                return None

        while True:
            item = await asyncio.to_thread(next_item)
            if item is None:
                break
            wav_chunk, sr, _ = item
            buf = io.BytesIO()
            sf.write(buf, np.array(wav_chunk), sr, format="WAV")
            yield buf.getvalue()

    # ------------------------------------------------------------------
    # Batch form
    # ------------------------------------------------------------------

    def _generate_full(self, text, language, voice_path, temperature) -> bytes:
        wav_arrays, sample_rate = self._model.generate_voice_clone(
            text=text, language=language, ref_audio=voice_path,
            temperature=temperature, xvec_only=True,
        )
        wav = np.concatenate(wav_arrays) if isinstance(wav_arrays, list) else wav_arrays
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate, format="WAV")
        result = buf.getvalue()
        del wav, wav_arrays
        return result
