import os
import logging
from typing import List
import numpy as np

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
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer
                
                logger.info(f"BgeMicro: Loading model {self.model_id} on {self.backend}")
                self.model = ORTModelForFeatureExtraction.from_pretrained(self.model_id, provider="CPUExecutionProvider")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                logger.info("BgeMicro loaded successfully.")
            except Exception as e:
                logger.error(f"BgeMicro: Failed to load: {e}")
                raise e

    def run(self, input_text: str) -> List[float]:
        if self.model is None:
            self.load()
            
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            # Perform pooling (usually mean pooling for BGE)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Ensure it's a list of floats
            result = embeddings[0].detach().cpu().numpy().tolist()
            return result
        except Exception as e:
            logger.error(f"BgeMicro: Execution failed: {e}")
            raise e
