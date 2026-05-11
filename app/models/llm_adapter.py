import logging
from typing import Optional, Dict, Any

from .base_model import BaseModel
from ..services.llm_api import llm_api

logger = logging.getLogger("JulyEngine.Models.BaseModel")


class LLMAdapter(BaseModel):
    def __init__(self, model_meta=None):
        super().__init__('api', model_meta)
        self._llm = llm_api
        
    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        pass        

    async def run(self, payload: Dict[str, Any], **kwargs):
        return await self._llm.run_chat(payload)

    def unload(self, model_name: Optional[str] = None):
        pass

    def is_loaded(self):
        return True