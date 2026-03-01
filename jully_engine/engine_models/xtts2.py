import os
import logging
from typing import Any, Dict, Optional
import torch
from TTS.api import TTS

logger = logging.getLogger("JulyEngine.Models.XTTS2")

class XTTS2:
    """
    XTTS v2 model for high-quality TTS.
    Backends: cpu, gpu.
    """
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cuda" if backend == "gpu" and torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            try:
                logger.info(f"XTTS2: Loading model on {self.device}")
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                logger.info("XTTS2 loaded successfully.")
            except Exception as e:
                logger.error(f"XTTS2: Failed to load: {e}")
                raise e

    def run(self, text: str, voice_path: str, language: str, output_path: str) -> str:
        if self.model is None:
            self.load()
            
        try:
            logger.info(f"XTTS2: Synthesizing to {output_path} using speaker {voice_path}")
            self.model.tts_to_file(
                text=text,
                speaker_wav=voice_path,
                language=language,
                file_path=output_path
            )
            return output_path
        except Exception as e:
            logger.error(f"XTTS2: Execution failed: {e}")
            raise e
