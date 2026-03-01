import logging
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
        if self.model_tag == "faster-whisper":
            return FasterWhisper(backend=self.backend)
        elif self.backend == "api":
            return LLMApi(backend=self.backend)
        else:
            raise ValueError(f"Ears: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def listen(self, audio_data: bytes, language: Optional[str] = None):
        if isinstance(self._strategy, FasterWhisper):
            return self._strategy.run(audio_data, language=language)
        elif isinstance(self._strategy, LLMApi):
            # litellm transcription support
            import io
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav" # litellm/OpenAI requires a filename
            base_url = None # can be passed from payload headers
            return self._strategy.run_stt(self.model_tag, audio_file, base_url=base_url)
        return None
