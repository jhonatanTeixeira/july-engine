import logging
from typing import Any, Dict, List, Optional
from ..engine_models.bge_micro import BgeMicro
from ..engine_models.multilingual_e5 import MultilingualE5
from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Memory")

class Memory:
    """
    Handles embeddings for long-term memory.
    Strategies: BgeMicro (cpu), MultilingualE5 (gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        elif self.model_tag == "bge-micro":
            return BgeMicro(backend="cpu")
        elif self.model_tag == "multilingual-e5":
            return MultilingualE5(backend="gpu")
        else:
            raise ValueError(f"Memory: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def embed(self, payload: Dict[str, Any]):
        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            input_text = payload.pop("input", "")
            headers = payload.pop("headers", {})
            return self._strategy.run_embeddings(model, input_text, headers=headers, **payload)
            
        elif isinstance(self._strategy, (BgeMicro, MultilingualE5)):
            input_text = payload.get("input", "")
            return self._strategy.run(input_text)
            
        return None
