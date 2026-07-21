import io
import logging
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.F5TTS")

DEFAULT_MODEL = "F5TTS_v1_Base"


class F5TTSModel(BaseModel):
    """
    F5-TTS (SWivid) — ~336M param DiT flow-matching TTS, fast non-autoregressive
    zero-shot voice cloning from a short reference clip.
    """

    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.model_variant = self.meta.get("variant") or self.meta.get("model") or DEFAULT_MODEL
        self.device = "cuda" if backend == "gpu" else "cpu"
        self._model = None
        self._ref_text_cache: Dict[str, str] = {}

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 3000

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return

        from f5_tts.api import F5TTS

        logger.info(f"F5TTS: Loading {self.model_variant} on {self.device}")
        self._model = F5TTS(model=self.model_variant, device=self.device)
        logger.info("F5TTS loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        if self._model is not None:
            del self._model
            self._model = None

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    async def run(self, payload: Dict[str, Any], **kwargs) -> bytes:
        from ..services.voice_service import voice_service

        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "default")
        seed = payload.get("seed")

        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            raise ValueError(f"F5TTS: Voice '{voice_id}' not found")
        voice_path, _ = voice_res

        if self._model is None:
            self.load()

        ref_text = await self._resolve_ref_text(voice_id, voice_path, payload.get("ref_text"))

        wav, sr, _spec = self._model.infer(
            ref_file=voice_path,
            ref_text=ref_text,
            gen_text=text,
            speed=float(payload.get("speed", 1.0)),
            nfe_step=int(payload.get("nfe_step", 32)),
            cfg_strength=float(payload.get("cfg_strength", 2.0)),
            seed=seed,
        )

        buf = io.BytesIO()
        sf.write(buf, np.array(wav), sr, format="WAV")
        return buf.getvalue()

    async def _resolve_ref_text(self, voice_id: str, voice_path: str, provided: Optional[str]) -> str:
        if provided:
            return provided
        if voice_id in self._ref_text_cache:
            return self._ref_text_cache[voice_id]

        # F5-TTS needs the reference clip's transcript; auto-transcribe via the
        # engine's own STT pipeline when not supplied (cached per voice).
        from ..bridge import bridge

        with open(voice_path, "rb") as f:
            audio_bytes = f.read()

        text = await bridge.process_stt({"audio": audio_bytes}, {})
        text = text if isinstance(text, str) else str(text)
        self._ref_text_cache[voice_id] = text
        return text
