import os
import logging
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.FasterWhisper")


class FasterWhisperModel(BaseModel):
    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.device = "cpu"
        self._model = None
        self.model_size = os.environ.get("STT_MODEL", "medium")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu":
            return 0
        return 1000

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return
        import torch
        from faster_whisper import WhisperModel

        self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        logger.info(f"FasterWhisper: Loading {self.model_size} on {self.device}")
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
        logger.info("FasterWhisper loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        if self._model:
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

    def run(self, payload: Dict[str, Any], **kwargs) -> str:
        audio_data = payload.get("audio") or payload.get("audio_data") or payload.get("file")
        language = payload.get("language")

        if self._model is None:
            self.load()

        import io
        import numpy as np
        import soundfile as sf
        import noisereduce as nr

        audio_io = io.BytesIO(audio_data)
        data, rate = sf.read(audio_io)

        if len(data.shape) > 1:
            data = data.mean(axis=1)

        reduced = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)

        clean_io = io.BytesIO()
        sf.write(clean_io, reduced, rate, format="WAV", subtype="PCM_16")
        clean_io.seek(0)

        segments, _ = self._model.transcribe(
            clean_io,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return " ".join(s.text for s in segments).strip()
