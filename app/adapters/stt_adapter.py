from typing import Dict, Any, Optional

from .adapter_base import AdapterBase


class STTAdapter(AdapterBase):
    def __init__(self, task_type: str, backend="gpu", model_meta=None):
        super().__init__(task_type, backend, model_meta)
        self.model = None

    @classmethod
    def get_engine_type(cls, task_type):
        return "STT"

    def _ensure_model_instance(self):
        if self.model is None:
            from ..models.faster_whisper import FasterWhisperModel
            self.model = FasterWhisperModel(self.backend, self.meta)
        return self.model

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        # Instancia a classe do modelo (leve) para obter a estimativa real baseada no tamanho
        model = self._ensure_model_instance()
        return await model.get_required_vram(payload)

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        model = self._ensure_model_instance()
        model.load(n_ctx=n_ctx, num_layers=num_layers)

    async def run(self, payload: Dict[str, Any], **kwargs):
        if not self.is_loaded():
            self.load()
            
        return self.model.run(payload)

    def unload(self, model_name: Optional[str] = None):
        if self.model:
            self.model.unload()

    def is_loaded(self):
        return self.model is not None and self.model.is_loaded()
