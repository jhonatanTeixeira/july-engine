from __future__ import annotations
import os
import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
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
                from huggingface_hub import hf_hub_download
                
                logger.info(f"BgeMicro: Verifying/Downloading model {self.model_id}...")
                
                # O BGE Small ONNX geralmente está no repo 'optimum/all-MiniLM-L6-v2' 
                # ou o próprio usuário exportou. Vamos assumir que buscamos um .onnx padrão.
                # Se o repo oficial não tiver .onnx, usamos um mirror comum para Optimum.
                model_path = hf_hub_download(repo_id=self.model_id, filename="onnx/model.onnx")
                
                logger.info(f"BgeMicro: Loading ONNX session from {model_path}")
                self.model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                logger.info("BgeMicro loaded successfully.")
            except Exception as e:
                logger.error(f"BgeMicro: Failed to load: {e}")
                # Fallback ou erro explícito
                raise e
        

    # O caller passa is_query=True para buscar, ou is_query=False para gravar
    def run(self, input_text: str) -> List[float]:
        if self.model is None:
            self.load()
            
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        try:
            inputs = self.tokenizer(input_text, return_tensors="np", padding=True, truncation=True, max_length=512)
            # ONNX Runtime expects a dictionary of {input_name: numpy_array}
            ort_inputs = {k: v for k, v in inputs.items()}
            outputs = self.model.run(None, ort_inputs)
            
            # 1. CLS Pooling (pega o token [CLS] na posição 0 do primeiro output)
            sentence_embeddings = outputs[0][:, 0]
            
            # 2. L2 Normalization (usando numpy já que o output do ONNX é numpy)
            import numpy as np
            norm = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
            sentence_embeddings = sentence_embeddings / norm
            
            result = sentence_embeddings[0].tolist()
            logger.info(f"Engine BgeMicro executed successfully on {self.backend} with {self.model_id}")
            return result
            
        except Exception as e:
            logger.error(f"BgeMicro: Execution failed: {e}")
            raise e

    def run_passage(self, input_text: str) -> List[float]:
        return self.run(input_text)

    def run_query(self, input_text: str) -> List[float]:
        return self.run(input_text)