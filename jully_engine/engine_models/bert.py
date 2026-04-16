import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("JulyEngine.Models.Bert")

class BertEmbedder:
    """
    Base class for BERT-based embedders (CodeBERT, GraphCodeBERT).
    """
    def __init__(self, model_id: str, backend: str = "gpu"):
        self.model_id = model_id
        self.backend = backend
        self.device = "cpu"
        self.model = None
        self.tokenizer = None

    def is_loaded(self) -> bool:
        return self.model is not None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu":
            return 0
        # CodeBERT/GraphCodeBERT base are ~500MB, plus activation and some overhead
        return 800 

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
            
            try:
                logger.info(f"BertEmbedder: Loading model {self.model_id} on {self.device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
                self.model.eval()
                logger.info(f"BertEmbedder: {self.model_id} loaded successfully.")
            except Exception as e:
                logger.error(f"BertEmbedder: Failed to load {self.model_id}: {e}")
                raise e

    def run(self, input_text: str) -> List[float]:
        if self.model is None:
            self.load()

        import torch
        import torch.nn.functional as F

        try:
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Mean Pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            sentence_embeddings = sum_embeddings / sum_mask
            
            # L2 Normalization
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            result = sentence_embeddings[0].cpu().tolist()
            logger.debug(f"BertEmbedder: Execution successful for {self.model_id}")
            return result
            
        except Exception as e:
            logger.error(f"BertEmbedder: Execution failed for {self.model_id}: {e}")
            raise e

    def run_passage(self, input_text: str) -> List[float]:
        return self.run(input_text)

    def run_query(self, input_text: str) -> List[float]:
        return self.run(input_text)


class CodeBERT(BertEmbedder):
    def __init__(self, backend: str = "gpu"):
        super().__init__("microsoft/codebert-base", backend)


class CodeGraphBERT(BertEmbedder):
    def __init__(self, backend: str = "gpu"):
        super().__init__("microsoft/graphcodebert-base", backend)
