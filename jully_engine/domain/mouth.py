from __future__ import annotations
import logging
import os
import json
import inspect
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
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
                from ..engine_models.replicate_api import Replicate
                return Replicate()
            
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)
        elif self.model_tag == "xtts":
            from ..engine_models.xtts2 import XTTS2
            return XTTS2(backend=self.backend)
        elif self.model_tag == "piper":
            from ..engine_models.piper import Piper
            return Piper(backend=self.backend)
        elif self.model_tag.startswith("kokoro"):
            from ..engine_models.kokoro_tts import KokoroTTS
            return KokoroTTS(backend=self.backend, model_tag=self.model_tag)
        else:
            raise ValueError(f"Mouth: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            return self._strategy.get_required_vram(payload)
        return 0

    async def speak(self, payload: Dict[str, Any]) -> Optional[bytes]:
        from ..engine_models.replicate_api import Replicate
        from ..engine_models.xtts2 import XTTS2
        from ..engine_models.piper import Piper
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.kokoro_tts import KokoroTTS

        # For local strategies, unpack payload
        text = payload.get("input", payload.get("text", ""))
        
        headers: dict = payload.get("headers", {})
        
        from ..persistence import get_backend
        config = get_backend().get_setting("TTS") or {}
        
        voice_id = payload.get("voice", None) or config.get('voice', 'af_heart')
        language = payload.get("language", None) or config.get('language', 'a')

        if config:
            headers.setdefault('x-base-url', config.get('base_url', None))
            headers.setdefault("x-api-key", config.get('api_key', None))
            headers.setdefault("authorization", f"Bearer {config.get('api_key', None)}")

        if isinstance(self._strategy, (LLMApi, Replicate)):
            model = payload.pop("model", self.model_tag)
            text = payload.pop("input", payload.pop("text", ""))
            voice_id = payload.pop("voice", voice_id)
            headers = payload.pop("headers", headers)

            audio_content = self._strategy.run_tts(model, text, voice_id, headers=headers, **payload)
            if inspect.iscoroutine(audio_content):
                audio_content = await audio_content
            return audio_content

        if isinstance(self._strategy, XTTS2):
            return self._strategy.run(text, voice_id, language)

        elif isinstance(self._strategy, Piper):
            return self._strategy.run(text, voice_id)

        elif isinstance(self._strategy, KokoroTTS):
            return await self._strategy.run(text, voice_id, language)

        return None

    def unload(self):
        """Libera os recursos da estratégia (XTTS2, Piper, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload()
            logger.info(f"Mouth: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Mouth: Strategy {self.model_tag} cleared.")
