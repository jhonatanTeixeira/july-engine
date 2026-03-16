import logging
import os
import json
from typing import Any, Dict, Optional

from ..engine_models.replicate_api import Replicate
from ..engine_models.xtts2 import XTTS2
from ..engine_models.piper import Piper
from ..engine_models.llm_api import LLMApi
from ..engine_models.kokoro_tts import KokoroTTS
from ..persistence import get_backend

logger = logging.getLogger("JulyEngine.Domain.Mouth")

class Mouth:
    """
    Handles Text-to-Speech (TTS) logic and voice resolution.
    Strategies: XTTS2 (cpu, gpu), Piper (cpu, gpu), KokoroTTS (cpu, gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()
        self.voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "voices"))
        self.persistence_backend = get_backend()

    def _get_strategy(self):
        if self.backend == "api":
            if self.model_tag.startswith('replicate/'):
                return Replicate()
            
            return LLMApi(backend=self.backend)
        elif self.model_tag == "xtts":
            return XTTS2(backend=self.backend)
        elif self.model_tag == "piper":
            return Piper(backend=self.backend)
        elif self.model_tag.startswith("kokoro"):
            return KokoroTTS(backend=self.backend, model_tag=self.model_tag)
        else:
            raise ValueError(f"Mouth: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def _resolve_voice(self, voice_id: str) -> Dict[str, Any]:
        """
        Resolves a voice ID to its configuration by checking uploaded voices from persistence.
        """
        uploaded_voices = self.persistence_backend.get_uploaded_voices()
        for v in uploaded_voices:
            if v.get("id") == voice_id:
                return v

        return {}

    async def speak(self, payload: Dict[str, Any]) -> Optional[bytes]:
        # For local strategies, unpack payload
        text = payload.get("input", payload.get("text", ""))
        voice_id = payload.get("voice", "af_heart")
        language = payload.get("language", "a")

        voice_info = self._resolve_voice(voice_id)

        # Override language if it was bound to the uploaded voice
        if voice_info.get("language"):
            language = voice_info.get("language")

        if isinstance(self._strategy, (LLMApi, Replicate)):
            model = payload.pop("model", self.model_tag)
            text = payload.pop("input", payload.pop("text", ""))
            voice_id = payload.pop("voice", voice_id)
            headers = payload.pop("headers", {})
            payload.setdefault('voice_info', voice_info)

            audio_content = self._strategy.run_tts(model, text, voice_id, headers=headers, **payload)
            return audio_content

        if isinstance(self._strategy, XTTS2):
            rel_path = voice_info.get("path")
            if not rel_path:
                logger.warning(f"Mouth(XTTS): Could not find valid 'path' for voice_id {voice_id}.")
                # fallback simple just to not break immediately
                full_voice_path = ""
            else:
                full_voice_path = os.path.join(self.voices_dir, rel_path)
            return self._strategy.run(text, full_voice_path, language)

        elif isinstance(self._strategy, Piper):
            # For Piper we use 'voice_id' as the model name if it's standard, 
            # or 'piper_path' if uploaded specifically. 
            hf_path = voice_info.get("piper_path")
            return self._strategy.run(text, voice_id, hf_path=hf_path)

        elif isinstance(self._strategy, KokoroTTS):
            return await self._strategy.run(text, voice_id, language)

        return None
