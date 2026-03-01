import os
import logging
from typing import List
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger("JulyEngine.Models.MultilingualE5")

class MultilingualE5:
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cuda" if backend == "gpu" and torch.cuda.is_available() else "cpu"
        self.model_id = "intfloat/multilingual-e5-small"
        self.model = None

    def load(self):
        if self.model is None:
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
            # E5 models usually expect "query: " or "passage: " prefix
            prefix = "query: " if len(input_text) < 512 else "passage: "
            embedding = self.model.encode(prefix + input_text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"MultilingualE5: Execution failed: {e}")
            raise e
