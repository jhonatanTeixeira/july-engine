import logging
from typing import Any, Dict, List, Optional
from ..engine_models.gguf import GGUF
from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Brain")

class Brain:
    """
    Handles text chat completions.
    Strategies: GGUF (cpu, gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        
        # Check if model is in DB or ends with .gguf
        from ..routers.models import load_models_db
        db = load_models_db()
        is_gguf = self.model_tag.endswith(".gguf") or self.model_tag in db
        
        if self.backend in ["gpu", "cpu"] and is_gguf:
            return GGUF(backend=self.backend, model_alias=self.model_tag)
        else:
            raise ValueError(f"Brain: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def chat(self, payload: Dict[str, Any]):
        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            headers = payload.pop("headers", {})
            return await self._strategy.run_chat(model, messages, stream=stream, headers=headers, **payload)
            
        elif isinstance(self._strategy, GGUF):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            return self._strategy.run_chat(messages, stream=stream, **payload)
