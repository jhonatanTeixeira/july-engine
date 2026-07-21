import gc
import logging
from typing import Any, Dict, List, Optional, Union

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.GLiNER2")

MODEL_ID = "fastino/gliner2-base-v1"


class GLiNER2Extractor(BaseModel):
    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.model_id = self.meta.get("model") or self.meta.get("alias") or MODEL_ID
        self._extractor = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 800

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if self._extractor is not None:
            return

        from gliner2 import GLiNER2

        # Quantize by default on CPU (no extra deps required per the library),
        # leave full precision on GPU unless explicitly requested via settings.
        quantize = self.meta.get("quantize")
        if quantize is None:
            quantize = self.backend == "cpu"

        logger.info(f"GLiNER2: Carregando {self.model_id} (backend={self.backend}, quantize={quantize})...")
        self._extractor = GLiNER2.from_pretrained(self.model_id, quantize=quantize)
        logger.info("GLiNER2: Carga finalizada com sucesso!")

    def is_loaded(self) -> bool:
        return self._extractor is not None

    def unload(self, model_name: Optional[str] = None):
        if self._extractor is None:
            return

        logger.info("GLiNER2: Descarregando modelo...")
        self._extractor = None
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def run(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self._extractor is None:
            self.load()

        text: Union[str, List[str]] = payload.get("text") or payload.get("input", "")
        labels: List[str] = payload.get("labels") or []
        threshold = float(payload.get("threshold", 0.0))
        include_confidence = bool(payload.get("include_confidence", True)) or threshold > 0
        include_spans = bool(payload.get("include_spans", True))

        if not labels:
            raise ValueError("GLiNER2Extractor: 'labels' is required (list of entity types to extract)")
        if not text:
            raise ValueError("GLiNER2Extractor: 'text' is required")

        # extract_entities/batch_extract_entities have no native confidence-threshold
        # kwarg (that only exists on the unrelated extract_json method), so filtering
        # by threshold is applied here as a post-processing step.
        if isinstance(text, list):
            result = self._extractor.batch_extract_entities(
                text,
                labels,
                include_confidence=include_confidence,
                include_spans=include_spans,
                batch_size=int(payload.get("batch_size", 8)),
            )
            if threshold > 0:
                return [self._filter_by_threshold(r, threshold) for r in result]
            return result

        result = self._extractor.extract_entities(
            text,
            labels,
            include_confidence=include_confidence,
            include_spans=include_spans,
        )
        if threshold > 0:
            return self._filter_by_threshold(result, threshold)
        return result

    @staticmethod
    def _filter_by_threshold(result: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        entities = result.get("entities", {})
        filtered = {
            label: [e for e in items if e.get("confidence", 1.0) >= threshold]
            for label, items in entities.items()
        }
        return {"entities": {label: items for label, items in filtered.items() if items}}
