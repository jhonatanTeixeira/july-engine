import logging
from typing import Any, Dict, List, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.MultilingualE5")

MODEL_ID = "intfloat/multilingual-e5-small"


class MultilingualE5Model(BaseModel):
    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.device = "cpu"
        self._model = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 500

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
        logger.info(f"MultilingualE5: Loading {MODEL_ID} on {self.device}")
        self._model = SentenceTransformer(MODEL_ID, device=self.device)
        logger.info("MultilingualE5 loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        self._model = None

    def run(self, payload: Dict[str, Any], **kwargs) -> List[float]:
        input_text = payload.get("input") or payload.get("text") or payload.get("query", "")
        emb_type = "passage" if payload.get("input") or payload.get("text") else "query"
        logger.info('MultilingualE5 is running')
        logger.debug('MultilingualE5 input: ' + str(input_text))

        if self._model is None:
            self.load()

        if emb_type == "query":
            input_text = f"query: {input_text}"
        elif emb_type == "passage":
            input_text = f"passage: {input_text}"
        else:
            raise ValueError("MultilingualE5 requires emb_type='query' or 'passage'")

        import torch
        with torch.inference_mode():
            embedding = self._model.encode(
                input_text,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        return embedding.tolist()
