from __future__ import annotations
import os
import logging
from typing import Optional, TYPE_CHECKING
from io import BytesIO

if TYPE_CHECKING:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

logger = logging.getLogger("JulyEngine.Models.KokoroTTS")

class KokoroTTS:
    def __init__(self, backend="cpu", model_tag="kokoro"):
        self.backend = backend
        self.model_tag = model_tag
        self.pipeline = None
        self.device = "cuda" if backend == "gpu" else "cpu"
        self.lang_code = None

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o Kokoro TTS."""
        if self.device == "cpu":
            return 0
        return 500 # ~500MB

    def load(self, lang_code='a'):
        if self.pipeline is None or lang_code != self.lang_code:
            from kokoro import KPipeline
            try:
                logger.info(f"KokoroTTS: Loading model on {self.device}")
                # You typically specify language code 'a' for American English, etc.
                self.pipeline = KPipeline(lang_code=lang_code, device=self.device)
                self.lang_code = lang_code
                logger.info("KokoroTTS loaded successfully.")
            except Exception as e:
                logger.error(f"KokoroTTS: Failed to load: {e}")
                raise e

    def clear(self):
        if self.pipeline is not None:
            logger.info("KokoroTTS: Unloading model")
            del self.pipeline
            self.pipeline = None

    async def run(self, text: str, voice_id: str, lang_code: str) -> bytes:
        self.load(lang_code)
            
        try:
            # generate returns a generator of (graphemes, phonemes, audio_array)
            generator = self.pipeline(text, voice=voice_id, speed=1.0, split_pattern=r'\n+')
            
            audio_segments = []
            
            for _, _, audio in generator:
                audio_segments.append(audio)
            
            if not audio_segments:
                raise ValueError("No audio generated")
                
            import numpy as np
            final_audio = np.concatenate(audio_segments)
            
            import soundfile as sf
            # Convert to WAV bytes in memory
            buffer = BytesIO()
            # Kokoro sample rate is usually 24000
            sf.write(buffer, final_audio, 24000, format='WAV')
            logger.info(f"Engine KokoroTTS executed successfully on {self.backend} with {self.model_tag}")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"KokoroTTS execution failed: {e}")
            raise e
