import logging
from typing import Any, Dict, List, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.BgeMicro")

MODEL_ID = "BAAI/bge-small-en-v1.5"


class BgeMicroModel(BaseModel):
    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._model = None
        self._tokenizer = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        logger.info(f"BgeMicro: Loading {MODEL_ID}")
        model_path = hf_hub_download(repo_id=MODEL_ID, filename="onnx/model.onnx")
        self._model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        logger.info("BgeMicro loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        self._model = None
        self._tokenizer = None

    def run(self, payload: Dict[str, Any], **kwargs) -> List[float]:
        input_text = payload.get("input") or payload.get("text") or payload.get("query", "")
        emb_type = payload.get("emb_type", "default")

        if self._model is None:
            self.load()

        if emb_type == "query":
            return self._embed(input_text)
        if emb_type == "passage":
            return self._embed(input_text)
        return self._embed(input_text)

    def _embed(self, text: str) -> List[float]:
        import numpy as np

        inputs = self._tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        outputs = self._model.run(None, dict(inputs))
        embeddings = outputs[0][:, 0]
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norm)[0].tolist()
