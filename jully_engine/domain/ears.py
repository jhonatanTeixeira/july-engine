from __future__ import annotations
import logging
import io
import inspect
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_models.faster_whisper import FasterWhisper
    from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Ears")

class Ears:
    """
    Handles Speech-to-Text.
    Strategies: FasterWhisper (cpu, gpu), LLMApi (api).
    Contract: listen() ALWAYS returns a pure string (str).
    """
    def _get_strategy(self):
        if self.backend == "api":
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)
        elif self.backend in ["gpu", "cpu"]:
            from ..engine_models.faster_whisper import FasterWhisper
            return FasterWhisper(backend=self.backend)
        else:
            raise ValueError(f"Ears: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            return self._strategy.get_required_vram(payload)
        return 0

    def _extract_text(self, response: Any) -> str:
        """Helper to force OpenAI-like STT responses into pure text."""
        if not response:
            return ""
            
        # Padrão OpenAI (dict com a chave 'text')
        if isinstance(response, dict):
            return response.get("text", str(response))
            
        # Caso o LiteLLM retorne um objeto Pydantic/Namespace com o atributo 'text'
        if hasattr(response, "text"):
            return str(response.text)
            
        # Fallback para string pura (como o FasterWhisper já faz)
        return str(response)

    async def listen(self, audio_data: bytes, language: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> str:
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.faster_whisper import FasterWhisper

        if payload is None:
            payload = {}
            
        headers = payload.get("headers", {})
        
        from ..persistence import get_backend
        config = get_backend().get_setting("STT")
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"
            
        # 1. Rota API (Requer formatação do arquivo e chamada assíncrona)
        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            headers = payload.pop("headers", headers)
            
            # litellm transcription support requires a file-like object with a name
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav" 
            
            if language and "language" not in payload:
                payload["language"] = language
                
            # Correção do Bug: Chamada com 'await' para a API assíncrona
            raw_response = await self._strategy.run_stt(model, audio_file, headers=headers, **payload)
            return self._extract_text(raw_response)
            
        # 2. Rota Local (FasterWhisper)
        elif isinstance(self._strategy, FasterWhisper):
            # Assumindo que FasterWhisper.run é síncrono. 
            # Caso contrário, precisamos adaptar para 'await' aqui também.
            raw_response = self._strategy.run(audio_data, language=language)
            
            # Se por acaso a implementação local for assíncrona, capturamos aqui:
            if inspect.iscoroutine(raw_response):
                raw_response = await raw_response
                
            return self._extract_text(raw_response)
            
        return ""

    def is_loaded(self):
        return hasattr(self._strategy, "is_loaded") and self._strategy.is_loaded()

    def load(self):
        if hasattr(self._strategy, "load"):
            self._strategy.load()


    def unload(self):
        """Libera os recursos da estratégia"""
        if hasattr(self._strategy, "unload"):
            # Alguns unloads pedem o model_tag (como no Eyes), mas deixei vazio como no original
            try:
                self._strategy.unload()
            except TypeError:
                self._strategy.unload(self.model_tag)
            logger.info(f"Ears: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Ears: Strategy {self.model_tag} cleared.")