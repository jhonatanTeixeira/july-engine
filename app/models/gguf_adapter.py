import asyncio
import logging
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.GGUFAdapter")


class GGUFAdapter(BaseModel):
    """
    Thin adapter around the shared llama_gguf.GGUF library.
    Exposes the interface expected by LLMModel without coupling
    the rest of the codebase to the llama_gguf internals.
    """
    def __init__(self, backend: str, model_meta: dict):
        from llama_gguf import GGUF
        
        # Garante compatibilidade de nomes de campos entre Studio e Lib
        if "model_id" not in model_meta and "id" in model_meta:
            model_meta["model_id"] = model_meta["id"]
            
        self._impl = GGUF(backend=backend, model=model_meta)
        self.backend = backend
        self.model_meta = model_meta

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if hasattr(self._impl, "get_required_vram"):
            res = self._impl.get_required_vram(payload)
            if asyncio.iscoroutine(res):
                return await res
            return res
        return 0

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if hasattr(self._impl, "load"):
            self._impl.load(n_ctx=n_ctx, num_layers=num_layers)

    def is_loaded(self) -> bool:
        return hasattr(self._impl, "is_loaded") and self._impl.is_loaded()

    def unload(self, model_name: Optional[str] = None):
        if hasattr(self._impl, "unload"):
            self._impl.unload(model_name or self.model_meta.get("alias"))

    def decrement_layers(self) -> bool:
        if hasattr(self._impl, "decrement_layers"):
            return self._impl.decrement_layers()
        return False

    async def run_chat(self, messages: list, stream: bool = False, **kwargs):
        return await self._impl.run_chat(messages, stream=stream, **kwargs)
    
    async def run(self, payload: Dict[str, Any], **kwargs):
        messages = payload.pop("messages", [])
        stream = payload.pop("stream", False)

        return await self.run_chat(messages, stream=stream, **kwargs, **payload)

