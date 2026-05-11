import io
import logging
import threading
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.XTTS2")

IDLE_TIMEOUT = 120


class XTTS2Model(BaseModel):
    """
    XTTS v2 TTS. No chunked streaming — returns full WAV bytes.
    Supports idle GPU→CPU offload after IDLE_TIMEOUT seconds.
    """

    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.device = "cpu"
        self._model = None
        self._idle_timer: Optional[threading.Timer] = None
        self._is_offloaded = False

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 2500

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return
        import torch
        from TTS.api import TTS

        self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
        logger.info(f"XTTS2: Loading on {self.device}")
        self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self._is_offloaded = False
        logger.info("XTTS2 loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        self._cancel_timer()
        if self._model:
            del self._model
            self._model = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("XTTS2 unloaded.")

    def run(self, payload: Dict[str, Any], **kwargs) -> bytes:
        from ..services.voice_service import voice_service

        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "default")
        language = payload.get("language", "en")
        temperature = payload.get("temperature", 0.7)

        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            raise ValueError(f"XTTS2: Voice '{voice_id}' not found")
        voice_path, voice_lang = voice_res
        target_lang = language or voice_lang

        if self._model is None:
            self.load()
        elif self._is_offloaded and self.device == "cuda":
            logger.info("XTTS2: Waking up from idle offload (CPU → GPU)")
            self._model = self._model.to(self.device)
            self._is_offloaded = False

        self._reset_timer()

        text = text.replace('"', "").replace("-", " ").replace(".", "\n").strip()
        if not text.endswith((".", "!", "?")):
            text += "."

        wav = self._model.tts(text=text, speaker_wav=voice_path, language=target_lang, temperature=temperature)

        buf = io.BytesIO()
        sf.write(buf, np.array(wav), 24000, format="WAV")
        return buf.getvalue()

    # ------------------------------------------------------------------

    def _reset_timer(self):
        if self.device == "cpu":
            return
        self._cancel_timer()
        self._idle_timer = threading.Timer(IDLE_TIMEOUT, self._idle_offload)
        self._idle_timer.start()

    def _cancel_timer(self):
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _idle_offload(self):
        if self._model and not self._is_offloaded:
            logger.info("XTTS2: Idle timeout — offloading GPU → CPU")
            self._model = self._model.to("cpu")
            self._is_offloaded = True
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
