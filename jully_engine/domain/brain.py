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
        
        from ..persistence import get_backend
        backend_db = get_backend()
        text_presets = backend_db.get_setting("TEXT_PRESETS") or []
        config = next((p for p in text_presets if p.get("alias") == self.model_tag), None)
        if not config and text_presets:
            config = text_presets[0]
            
        mcp_option = "emulated"
        
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"

            if config.get("reasoning_enabled"):
                payload["reasoning_enabled"] = config.get("reasoning_enabled")
                payload["reasoning_effort"] = config.get("reasoning_effort", "medium")
                
            payload['model'] = config.get('model', self.model_tag)
                
            mcp_option = config.get("mcp_option", "emulated")

        enable_internal_mcp = headers.get("x-enable-internal-mcp", "0") == "1"
        tools_whitelist = payload.get("tools_whitelist")
        mcp_handler = None
        
        if enable_internal_mcp:
            if mcp_option == "emulated":
                mcp_handler = self._mcp
            elif mcp_option == "internal":
                from ..services.internal_mcp import InternalMCP
                mcp_handler = InternalMCP()
            elif mcp_option == "external_only":
                from ..services.external_mcp import external_mcp_manager
                mcp_handler = external_mcp_manager
            
        if mcp_handler:
            mcp_handler.inject_tools(payload, tools_whitelist)
            
            if mcp_option == "emulated":
                payload.pop("tools", None)

        # Salva o payload com a injeção do XML já feita para o segundo turno (ReAct)
        import copy
        original_payload = copy.deepcopy(payload)

        # Filtra imagens e áudios pois o Brain é puro texto
        for msg in payload.get("messages", []):
            if isinstance(msg.get("content"), list):
                msg["content"] = [item for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"]

        if isinstance(self._strategy, LLMApi):
            model = payload.pop('model', self.model_tag)
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            req_headers = payload.pop("headers", {})
            response = await self._strategy.run_chat(model, messages, stream=stream, headers=req_headers, **payload)
            
        elif isinstance(self._strategy, GGUF):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            response = self._strategy.run_chat(messages, stream=stream, **payload)

        if mcp_handler:
            return await mcp_handler.orchestrate(response, self, original_payload)
                
        return response