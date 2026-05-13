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
        if self.backend == "api":
            from ..services.llm_api import llm_api
            # Inject config from settings if available
            config = self._load_stt_config()
            headers = payload.get("headers", {})
            if config:
                headers.setdefault("x-base-url", config.get("base_url"))
                headers.setdefault("x-api-key", config.get("api_key"))
            
            p = {
                **payload,
                "model": payload.get("model") or config.get("model") or self.model_id,
                "headers": headers
            }
            return await llm_api.dispatch("stt", p)

        if not self.is_loaded():
            self.load()
            
        return self.model.run(payload)

    def _load_stt_config(self) -> dict:
        try:
            from ..services.models_service import model_service
            return model_service.backend.get_setting("STT") or {}
        except Exception:
            return {}

    def unload(self, model_name: Optional[str] = None):
        if self.model:
            self.model.unload()

    def is_loaded(self):
        return self.model is not None and self.model.is_loaded()
