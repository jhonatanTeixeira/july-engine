import os
import logging

# os.environ["CT2_CUDA_ALLOCATOR"] = "cudaMalloc"

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
            
    def unload(self):
        del self.model
        self.model = None

    def run(self, audio_data: bytes, language: Optional[str] = None) -> str:
        if self.model is None:
            self.load()
            
        try:
            import io
            import numpy as np
            import soundfile as sf
            import noisereduce as nr
            
            # 1. Read the audio bytes into a numpy array
            audio_io = io.BytesIO(audio_data)
            data, rate = sf.read(audio_io)

            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # 2. Apply advanced background noise reduction
            logger.info("FasterWhisper: Applying background noise reduction...")
            reduced_noise_audio = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)

            # 3. Write the cleaned audio back to a bytes buffer in WAV format
            clean_audio_io = io.BytesIO()
            sf.write(clean_audio_io, reduced_noise_audio, rate, format='WAV', subtype='PCM_16')
            clean_audio_io.seek(0)
            
            # 4. Transcribe the cleaned audio (with built-in vad_filter to drop empty silences)
            logger.info("FasterWhisper: Transcribing cleaned audio...")
            segments, info = self.model.transcribe(clean_audio_io, language=language, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
            text = " ".join([segment.text for segment in segments]).strip()
            logger.info(f"Engine FasterWhisper executed successfully on {self.backend} with {self.model_size}")
            return text
        except Exception as e:
            logger.error(f"FasterWhisper: Transcription failed: {e}")
            raise e
