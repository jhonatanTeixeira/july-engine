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

    # O caller passa is_query=True para buscar, ou is_query=False para gravar
    def run(self, input_text: str) -> List[float]:
        if self.model is None:
            self.load()
            
        try:
            if not input_text.startswith('query: ') and not input_text.startswith('passage: '):
                raise Exception('must use query for serach embbedings or passage for persisting embeddings')
            
            # E5 usa a lib SentenceTransformer que já faz pooling e normalização se pedirmos
            embedding = self.model.encode(input_text, normalize_embeddings=True)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"MultilingualE5: Execution failed: {e}")
            raise e