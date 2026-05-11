import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("JulyEngine.Models.BaseModel")


class BaseModel:
    def __init__(self, backend="gpu", model_meta=None):
        self.backend = backend
        self.meta = model_meta or {}
        # Tenta obter ID de várias formas comuns
        self.model_id = self.meta.get("model") or self.meta.get("alias")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        pass

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        pass

    def run(self, payload: Dict[str, Any], **kwargs):
        pass

    def unload(self, model_name: Optional[str] = None):
        pass

    def is_loaded(self):
        pass