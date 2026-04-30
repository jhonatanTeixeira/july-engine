import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("JulyEngine.Models.MultilingualE5")

class MultilingualE5:
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cpu" # Default until load()
        self.model_id = "intfloat/multilingual-e5-small"
        self.model = None

    def is_loaded(self):
        return self.model is not None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o MultilingualE5."""
        if self.backend == "cpu":
            return 0
        return 500 # ~500MB

    def load(self):
        if self.model is None:
            import torch
            self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
            from sentence_transformers import SentenceTransformer
            try:
                logger.info(f"MultilingualE5: Loading model {self.model_id} on {self.device}")
                self.model = SentenceTransformer(self.model_id, device=self.device)
                logger.info("MultilingualE5 loaded successfully.")
            except Exception as e:
                logger.error(f"MultilingualE5: Failed to load: {e}")
                raise e

    def run(self, input_text: str) -> List[float]:
        if self.model is None:
            self.load()
            
        try:
            if not input_text.startswith('query: ') and not input_text.startswith('passage: '):
                raise Exception('must use query for search embbedings or passage for persisting embeddings')
            
            # E5 usa a lib SentenceTransformer que já faz pooling e normalização se pedirmos
            embedding = self.model.encode(input_text, normalize_embeddings=True)
            
            logger.info(f"Engine MultilingualE5 executed successfully on {self.backend} with {self.model_id}")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"MultilingualE5: Execution failed: {e}")
            raise e

    def run_passage(self, input_text: str) -> List[float]:
        return self.run('passage: ' + input_text)

    def run_query(self, input_text: str) -> List[float]:
        return self.run('query: ' + input_text)