from typing import Dict, Any, Optional

from ..models.base_model import BaseModel

class STTAdapter(BaseModel):
    def __init__(self, backend="gpu", model_meta=None):
        super().__init__(backend, model_meta)
        self.model = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        self.model.get_required_vram(payload)

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        from ..models.faster_whisper import FasterWhisperModel
        self.model = FasterWhisperModel(self.backend, self.meta)

    def run(self, payload: Dict[str, Any], **kwargs):
        if not self.model:
            self.load()
            
        return self.model.run(payload)

    def unload(self, model_name: Optional[str] = None):
        if self.model:
            self.model.unload()

    def is_loaded(self):
        return self.model and self.model.is_loaded()
    