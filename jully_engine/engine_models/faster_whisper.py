import os
import logging
import torch
from typing import Any, Dict, Optional
from faster_whisper import WhisperModel

logger = logging.getLogger("JulyEngine.Models.FasterWhisper")

class FasterWhisper:
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cuda" if backend == "gpu" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_size = os.environ.get("STT_MODEL", "medium")

    def load(self):
        if self.model is None:
            try:
                logger.info(f"FasterWhisper: Loading model {self.model_size} on {self.device}")
                # For GPU, usually use float16, for CPU int8 or float32
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
                logger.info("FasterWhisper loaded successfully.")
            except Exception as e:
                logger.error(f"FasterWhisper: Failed to load: {e}")
                raise e

    def run(self, audio_data: bytes, language: Optional[str] = None) -> str:
        if self.model is None:
            self.load()
            
        try:
            import io
            segments, info = self.model.transcribe(io.BytesIO(audio_data), language=language)
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"FasterWhisper: Transcription failed: {e}")
            raise e
