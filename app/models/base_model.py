import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("JulyEngine.Models.BaseModel")


class BaseModel:
    def __init__(self, backend="gpu", model_meta=None):
        self.backend = backend
        self.meta: dict = model_meta

        if not self.meta:
            from ..services.models_service import model_service
            engine_type = self.get_engine_type()
            if engine_type:
                self.meta = model_service.backend.get_setting(engine_type)

            # Garante que meta seja pelo menos um dicionário vazio
            if self.meta is None:
                self.meta = {}

        self.model_id: str = self.meta.get("model") or self.meta.get("alias")
        
    @classmethod
    def get_engine_type(cls):
        return None

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