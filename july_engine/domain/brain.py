import logging
import inspect
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_models.llama_gguf import GGUF
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
        self.config = None
        self._strategy = self._get_strategy()
        self._mcp = McpEmulator(InternalMCP())

    def _get_strategy(self):
        model_service = ModelsService()
        # Resolve by settings first to ensure presets (like mcp_option, template, etc.) are respected
        model = model_service.resolve_by_settings(self.model_tag) or model_service.get(self.model_tag)
        self.config = model

        logger.debug(f"Brain: Model {self.model_tag} loaded. Config: {self.config}")

        if self.backend == "api":
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend, model=model)
        
        if self.backend in ["gpu", "cpu"] and model is not None:
            from ..engine_models.llama_gguf import GGUF
            return GGUF(backend=self.backend, model=model)
        else:
            raise ValueError(f"Brain: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def is_loaded(self):
        return hasattr(self._strategy, "is_loaded") and self._strategy.is_loaded()

    def load(self):
        if hasattr(self._strategy, "load"):
            self._strategy.load()

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            res = self._strategy.get_required_vram(payload)
            if inspect.iscoroutine(res):
                return await res
            return res
        return 0

    async def chat(self, payload: Dict[str, Any]):
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.llama_gguf import GGUF
        
        mcp_option = "emulated"
        
        if self.config:
            mcp_option = self.config.get("mcp_option", "emulated")

        enable_internal_mcp = payload.get("headers", {}).get("x-enable-internal-mcp", "0") == "1"
        tools_whitelist = payload.pop("tools_whitelist", [])
        mcp_handler = None
        
        if enable_internal_mcp:
            if mcp_option == "internal":
                mcp_handler = self._mcp.internal_mcp
            elif mcp_option == "external_only":
                from ..services.external_mcp import external_mcp_manager
                mcp_handler = external_mcp_manager
            else: # emulated
                mcp_handler = self._mcp
        elif mcp_option == "emulated":
            # Special case: Enable McpEmulator only for XML/OpenAI conversion (no internal execution)
            mcp_handler = self._mcp
            
        if mcp_handler:
            mcp_handler.inject_tools(payload, tools_whitelist)
            
            if mcp_option == "emulated":
                payload.pop("tools", None)

            print(payload)

        # Salva o payload com a injeção do XML já feita para o segundo turno (ReAct)
        import copy
        original_payload = copy.deepcopy(payload)

        # Filtra imagens e áudios pois o Brain é puro texto (exceto se for um VLM nativo)
        is_vision = ModelsService().is_vision_model(self.model_tag)
        for msg in payload.get("messages", []):
            if isinstance(msg.get("content"), list):
                if not is_vision:
                    msg["content"] = [item for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"]
                else:
                    # Remove apenas áudio se for Vision
                    msg["content"] = [item for item in msg["content"] if isinstance(item, dict) and item.get("type") != "audio_url" and item.get("type") != "input_audio"]

        if isinstance(self._strategy, LLMApi):
            response = await self._strategy.run_chat(payload)
            
        elif isinstance(self._strategy, GGUF):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            response = await self._strategy.run_chat(messages, stream=stream, **payload)

        if mcp_handler:
            return await mcp_handler.orchestrate(response, self, original_payload)
                
        return response

    def unload(self):
        """Libera os recursos da estratégia (GGUF, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload(self.model_tag)
            logger.info(f"Brain: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Brain: Strategy {self.model_tag} cleared.")