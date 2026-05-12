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
        if self.backend == "gpu":
            return 100 # Pequeno custo para ONNX em GPU
        return 0

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        # Seleciona provedores ONNX baseados no backend
        providers = ["CPUExecutionProvider"]
        if self.backend == "gpu":
            # Tenta CUDA se disponível
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info(f"BgeMicro: Loading {MODEL_ID} on {self.backend} (Providers: {providers})")
        
        try:
            model_path = hf_hub_download(repo_id=MODEL_ID, filename="onnx/model.onnx")
            self._model = ort.InferenceSession(model_path, providers=providers)
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            logger.info("BgeMicro loaded successfully.")
        except Exception as e:
            logger.error(f"BgeMicro: Failed to load: {e}")
            if self.backend == "gpu":
                logger.info("BgeMicro: Retrying on CPU...")
                self._model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            else:
                raise e

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        self._model = None
        self._tokenizer = None
        logger.info("BgeMicro unloaded.")

    def run(self, payload: Dict[str, Any], **kwargs) -> List[float]:
        input_text = payload.get("input") or payload.get("text") or payload.get("query", "")
        
        if not input_text:
            return []

        if self._model is None:
            self.load()

        logger.info(f"BgeMicro: Running inference for text: '{input_text[:50]}...'")
        result = self._embed(input_text)
        return result

    def _embed(self, text: str) -> List[float]:
        import numpy as np

        inputs = self._tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        
        # ONNX expects inputs as a dict of {name: ndarray}
        # We need to filter inputs to match session input names
        session_inputs = {node.name: inputs[node.name] for node in self._model.get_inputs() if node.name in inputs}
        
        outputs = self._model.run(None, session_inputs)
        
        # BGE models normally use the [CLS] token (index 0)
        embeddings = outputs[0][:, 0]
        
        # L2 Normalization
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / (norm + 1e-9))[0].tolist()
