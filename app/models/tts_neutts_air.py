import io
import logging
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.NeuTTSAir")

DEFAULT_BACKBONE_REPO = "neuphonic/neutts-air"
DEFAULT_CODEC_REPO = "neuphonic/neucodec"
SAMPLE_RATE = 24000


class NeuTTSAirModel(BaseModel):
    """
    NeuTTS Air (Neuphonic) — ~0.7B Qwen2-based on-device TTS with instant
    voice cloning. CPU-first by design; also runs on GPU via backbone_device.
    """

    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.backbone_repo = self.meta.get("backbone_repo") or self.meta.get("model") or DEFAULT_BACKBONE_REPO
        self.codec_repo = self.meta.get("codec_repo") or DEFAULT_CODEC_REPO
        self.device = "cuda" if backend == "gpu" else "cpu"
        self._model = None
        self._ref_codes_cache: Dict[str, Any] = {}
        self._ref_text_cache: Dict[str, str] = {}

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 1500

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return

        from neutts import NeuTTS

        logger.info(f"NeuTTSAir: Loading {self.backbone_repo} on {self.device}")
        self._model = NeuTTS(
            backbone_repo=self.backbone_repo,
            backbone_device=self.device,
            codec_repo=self.codec_repo,
            codec_device=self.device,
        )
        logger.info("NeuTTSAir loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        if self._model is not None:
            del self._model
            self._model = None
        self._ref_codes_cache.clear()

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

        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            raise ValueError(f"NeuTTSAir: Voice '{voice_id}' not found")
        voice_path, _ = voice_res

        if self._model is None:
            self.load()

        ref_text = await self._resolve_ref_text(voice_id, voice_path, payload.get("ref_text"))
        ref_codes = self._resolve_ref_codes(voice_id, voice_path)

        wav = self._model.infer(text, ref_codes, ref_text)

        buf = io.BytesIO()
        sf.write(buf, np.array(wav), SAMPLE_RATE, format="WAV")
        return buf.getvalue()

    def _resolve_ref_codes(self, voice_id: str, voice_path: str):
        if voice_id not in self._ref_codes_cache:
            self._ref_codes_cache[voice_id] = self._model.encode_reference(voice_path)
        return self._ref_codes_cache[voice_id]

    async def _resolve_ref_text(self, voice_id: str, voice_path: str, provided: Optional[str]) -> str:
        if provided:
            return provided
        if voice_id in self._ref_text_cache:
            return self._ref_text_cache[voice_id]

        # NeuTTS Air needs the reference audio's transcript; auto-transcribe via
        # the engine's own STT pipeline (cached per voice so this only runs once).
        from ..bridge import bridge

        with open(voice_path, "rb") as f:
            audio_bytes = f.read()

        text = await bridge.process_stt({"audio": audio_bytes}, {})
        text = text if isinstance(text, str) else str(text)
        self._ref_text_cache[voice_id] = text
        return text
