import logging
import io
from typing import Any, Dict, Optional
from ..engine_models.faster_whisper import FasterWhisper
from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Ears")

class Ears:
    """
    Handles Speech-to-Text.
    Strategies: FasterWhisper (cpu, gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        elif self.backend in ["gpu", "cpu"]:
            return FasterWhisper(backend=self.backend)
        else:
            raise ValueError(f"Ears: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def listen(self, audio_data: bytes, language: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
        if payload is None:
            payload = {}
            
        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            headers = payload.pop("headers", {})
            
            # litellm transcription support
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav" # litellm/OpenAI requires a filename
            
            # language may have been passed separately
            if language and "language" not in payload:
                payload["language"] = language
                
            return self._strategy.run_stt(model, audio_file, headers=headers, **payload)
            
        elif isinstance(self._strategy, FasterWhisper):
            return self._strategy.run(audio_data, language=language)
            
        return None
