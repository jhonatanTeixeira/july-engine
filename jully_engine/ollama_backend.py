import ollama
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("OllamaBackend")

class OllamaBackend:
    def __init__(self, model: str = "llama3"):
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        logger.info(f"Ollama Chat with model {self.model}")
        response = ollama.chat(model=self.model, messages=messages)
        return response['message']['content']

    def generate(self, prompt: str, images: Optional[List[str]] = None) -> str:
        logger.info(f"Ollama Generate with model {self.model}")
        response = ollama.generate(model=self.model, prompt=prompt, images=images)
        return response['response']

    def embed(self, input: List[str]) -> List[List[float]]:
        logger.info(f"Ollama Embedding with model {self.model}")
        response = ollama.embed(model=self.model, input=input)
        return response['embeddings']

ollama_backend = OllamaBackend()
