import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("JulyEngine.Models.BaseModel")


class BaseModel:
    def __init__(self, backend="gpu", model_meta: dict | None = None):
        self.backend = backend
        
        self.meta: dict = model_meta or {}
        self.model_id: str = self.meta.get("alias") or self.meta.get("model") or ''
        
    @classmethod
    def get_engine_type(cls, task_type: str | None = None) -> str:
        raise NotImplementedError()

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        pass

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        pass

    async def run(self, payload: Dict[str, Any], **kwargs) -> Any:
        pass

    def unload(self, model_name: Optional[str] = None):
        pass

    def is_loaded(self):
        pass