import os
import logging
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("JulyEngine.Models.BgeMicro")

class BgeMicro:
    def __init__(self, backend="cpu"):
        self.backend = backend
        self.model_id = "BAAI/bge-small-en-v1.5"
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is None:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                
                logger.info(f"BgeMicro: Loading model {self.model_id} on {self.backend}")
                self.model = ort.InferenceSession(self.model_id, provider="CPUExecutionProvider")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                logger.info("BgeMicro loaded successfully.")
            except Exception as e:
                logger.error(f"BgeMicro: Failed to load: {e}")
                raise e

    # O caller passa is_query=True para buscar, ou is_query=False para gravar
    def run(self, input_text: str) -> List[float]:
        if self.model is None:
            self.load()
            
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            
            # 1. CLS Pooling (pega o token [CLS] na posição 0)
            sentence_embeddings = outputs.last_hidden_state[:, 0]
            
            # 2. L2 Normalization (Crítico para a distância de cosseno do BGE)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            result = sentence_embeddings[0].detach().cpu().numpy().tolist()
            logger.info(f"Engine BgeMicro executed successfully on {self.backend} with {self.model_id}")
            return result
            
        except Exception as e:
            logger.error(f"BgeMicro: Execution failed: {e}")
            raise e

    def run_passage(self, input_text: str) -> List[float]:
        return self.run(input_text)

    def run_query(self, input_text: str) -> List[float]:
        return self.run(input_text)