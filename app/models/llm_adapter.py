import asyncio
import logging
import asyncio
from typing import Optional, Dict, Any

from .base_model import BaseModel
from ..services.llm_api import llm_api

logger = logging.getLogger("JulyEngine.Models.BaseModel")


class LLMAdapter(BaseModel):
    def __init__(self, model_meta=None):
        super().__init__('api', model_meta)
        from ..services.llm_api import LLMApi
        # Usamos uma instância local configurada com os metadados do preset
        self._llm = LLMApi(backend='api', model=self.meta)
        
    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        pass        

    async def run(self, payload: Dict[str, Any], **kwargs):
        # Garante que o modelo no payload seja o modelo real (não o alias) se disponível no meta
        if "model" in self.meta:
            payload["model"] = self.meta["model"]
            
        reponse = await self._llm.run_chat(payload)

        if payload.get('stream'):
            async def response_iterator():
                async for chunk in reponse:
                    yield chunk.model_dump()
                    asyncio.sleep(0)

            return response_iterator()
        else:
            return reponse.model_dump()

    def unload(self, model_name: Optional[str] = None):
        pass

    def is_loaded(self):
        return True