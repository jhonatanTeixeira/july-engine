import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("JulyEngine.Models.XTTS2")

class XTTS2:
    """
    XTTS v2 model for high-quality TTS.
    Backends: cpu, gpu.
    """
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cpu" # Default until load()
        self.model = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o XTTS2 (~2.2GB)."""
        if self.backend == "cpu":
            return 0
        return 2200 # ~2.2GB para este modelo (XTTS2)

    def load(self):
        if self.model is None:
            import torch
            self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
            from TTS.api import TTS
            try:
                logger.info(f"XTTS2: Loading model on {self.device}")
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                logger.info("XTTS2 loaded successfully.")
            except Exception as e:
                logger.error(f"XTTS2: Failed to load: {e}")
                raise e

    def run(self, text: str, voice_id: str, language: str, temperature: float = 0.7) -> bytes:
        from ..services.voice_service import voice_service
        
        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            logger.error(f"XTTS2: Voice {voice_id} not found and no fallback available.")
            raise ValueError(f"Voice {voice_id} not found")
            
        voice_path, voice_lang = voice_res
        # Prioritize provided language, fallback to voice info language
        target_lang = language or voice_lang

        if self.model is None:
            self.load()

        text = text.replace(".", "").replace('"', '').replace("-", "")
            
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
                
            logger.info(f"XTTS2: Synthesizing to {output_path} using speaker {voice_path}")
            self.model.tts_to_file(
                text=text,
                speaker_wav=voice_path,
                language=language,
                file_path=output_path,
                temperature=temperature
            )
            
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
                
            os.remove(output_path)
            logger.info(f"Engine XTTS2 executed successfully on {self.backend} with XTTS2")
            return audio_bytes
        except Exception as e:
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"XTTS2: Execution failed: {e}")
            raise e

    def is_loaded(self):
        return self.model is not None

    def unload(self):
        """Libera os recursos da estratégia (XTTS2, Piper, etc)."""
        if self.model:
            del self.model
            self.model = None
            
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("XTTS2: Model unloaded.")
