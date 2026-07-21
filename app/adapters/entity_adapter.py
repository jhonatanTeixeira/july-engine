from typing import Any, Dict, Optional

from .adapter_base import AdapterBase


class EntityAdapter(AdapterBase):
    """
    Handles entity_extraction: zero-shot named-entity extraction via GLiNER2.

    Engine field: "entity_extraction"
    """

    def __init__(self, task_type: str, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(task_type, backend, model_meta)
        self.model = None

    @classmethod
    def get_engine_type(cls, task_type: str):
        return "ENTITY_EXTRACTION"

    def _ensure_model_instance(self):
        if self.model is None:
            from ..models.gliner2_extractor import GLiNER2Extractor
            self.model = GLiNER2Extractor(backend=self.backend, model_meta=self.meta)
        return self.model

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        model = self._ensure_model_instance()
        return await model.get_required_vram(payload)

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        model = self._ensure_model_instance()
        model.load(n_ctx=n_ctx, num_layers=num_layers)

    async def run(self, payload: Dict[str, Any], **kwargs):
        model = self._ensure_model_instance()

        # Fall back to the configured default label schema when the caller
        # doesn't send one, so a bare {"text": "..."} payload still works.
        if not payload.get("labels"):
            payload = {**payload, "labels": self.meta.get("labels") or _DEFAULT_LABELS}
        if payload.get("threshold") is None and self.meta.get("threshold") is not None:
            payload = {**payload, "threshold": self.meta.get("threshold")}

        if not self.is_loaded():
            self.load()

        return model.run(payload)

    def unload(self, model_name: Optional[str] = None):
        if self.model:
            self.model.unload()

    def is_loaded(self):
        return self.model is not None and self.model.is_loaded()


_DEFAULT_LABELS = [
    "person", "location", "organization", "event",
    "date", "emotion", "hobby", "animal", "food", "object",
]
