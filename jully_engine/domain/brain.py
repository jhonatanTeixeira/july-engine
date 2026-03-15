import logging
from typing import Any, Dict, List, Optional

from ..engine_models.gguf import GGUF
from ..engine_models.llm_api import LLMApi
from ..services.models_service import ModelsService
from ..services.mcp_emulator import McpEmulator
from ..services.internal_mcp import InternalMCP

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
        self._mcp = McpEmulator(InternalMCP())

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        
        model_service = ModelsService()
        model = model_service.get(self.model_tag) or model_service.resolve_by_settings(self.model_tag)
        
        if self.backend in ["gpu", "cpu"] and model is not None:
            return GGUF(backend=self.backend, model=model)
        else:
            raise ValueError(f"Brain: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def chat(self, payload: Dict[str, Any]):
        headers = payload.get("headers", {})
        enable_internal_mcp = headers.get("x-enable-internal-mcp", "0") == "1"
        
        if enable_internal_mcp:
            # 1. INJEÇÃO BARE-METAL: Injeta as tags XML nas 'messages' do payload
            self._mcp.inject_tools(payload)
            
            # 2. BYPASS DA API: Removemos o array 'tools' para impedir que o 
            # Llama.cpp tente formatar por conta própria e estrague o prompt.
            payload.pop("tools", None)

        # Salva o payload com a injeção do XML já feita para o segundo turno (ReAct)
        import copy
        original_payload = copy.deepcopy(payload)

        # Filtra imagens e áudios pois o Brain é puro texto
        for msg in payload.get("messages", []):
            if isinstance(msg.get("content"), list):
                msg["content"] = [item for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"]

        if isinstance(self._strategy, LLMApi):
            del payload["model"]
            model = self.model_tag
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            req_headers = payload.pop("headers", {})
            response = await self._strategy.run_chat(model, messages, stream=stream, headers=req_headers, **payload)
            
        elif isinstance(self._strategy, GGUF):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            response = self._strategy.run_chat(messages, stream=stream, **payload)

        if enable_internal_mcp:
            return await self._mcp.orchestrate(response, self, original_payload)
                
        return response