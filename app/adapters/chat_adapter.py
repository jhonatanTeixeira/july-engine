import copy
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from .adapter_base import AdapterBase
from ..models.base_model import BaseModel

if TYPE_CHECKING:
    from ..models.gguf_adapter import GGUFAdapter

logger = logging.getLogger("JulyEngine.Adapters.ChatAdapter")


class ChatAdapter(AdapterBase):
    """
    Adapter that unifies the internal XML tool calling logic (McpEmulator)
    with the local GGUF model execution.
    """

    def __init__(self, task_type: str, backend: str, model_meta: dict):
        super().__init__(task_type, backend, model_meta)

        from ..services.models_service import model_service

        if not (model := model_meta.get("model")):
            raise ValueError("no model defined for chat")
        
        self.meta = (model_service.backend.get_model(model) or {}) | model_meta

        self._strategy = None
        self._mcp = None

    @classmethod
    def get_engine_type(cls, task_type: str | None = None):
        return "TEXT_PRESETS"

    def _get_strategy(self) -> BaseModel:
        if not self._strategy:
            if self.backend in ['cpu', 'gpu']:
                from ..models.gguf_adapter import GGUFAdapter
                self._strategy = GGUFAdapter(self.backend, self.meta)
            else:
                from ..models.llm_adapter import LLMAdapter
                self._strategy = LLMAdapter(self.meta)

        return self._strategy

    def _get_mcp(self):
        # MCP externo removido - July Engine agora é 100% local
        # Se precisar de funcionalidade MCP, usar internal_mcp apenas via header x-enable-internal-mcp=1
        return None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return await self._get_strategy().get_required_vram(payload)

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        self._get_strategy().load(n_ctx=n_ctx, num_layers=num_layers)

    def is_loaded(self) -> bool:
        return self._strategy is not None and self._strategy.is_loaded()

    def unload(self, model_name: Optional[str] = None):
        if self._strategy:
            self._strategy.unload()

    async def chat(self, payload: dict):
        """Alias para run() usado pelo McpEmulator."""
        return await self.run(payload)

    async def run(self, payload: Dict[str, Any]):
        from ..models.helpers import MultiModalHelper

        helper = MultiModalHelper(payload)

        if not self.meta.get("is_vision"):
            await helper.process_vision()
        else:
            helper.sanitize_images()

        await helper.process_transcription()

        # MCP externo removido - July Engine agora é 100% local
        # TODO: internal_mcp pode ser movido para lugar separado se necessário
        
        original_payload = copy.deepcopy(payload)

        if not self.meta.get("is_vision"):
            payload["messages"] = helper.filter_images(payload.get("messages", []))
            payload["messages"] = helper.filter_audios(payload.get("messages", []))

        # Execute local model
        response = await self._get_strategy().run(payload)

        return self._ensure_string_content(response)

    def _ensure_string_content(self, data: Any) -> Any:
        """Garante que o campo content nunca seja None."""
        if isinstance(data, dict) and "choices" in data:
            for choice in data["choices"]:
                if "message" in choice and choice["message"].get("content") is None:
                    choice["message"]["content"] = ""
        return data