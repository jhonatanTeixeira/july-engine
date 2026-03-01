import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger("vLLMBackend")

class vLLMBackend:
    def __init__(self, model_name: str = "dphn/Dolphin3.0-Llama3.1-8B"):
        self.model_name = model_name
        self.llm = None
        self.sampling_params = None

    def _load_vllm(self):
        if self.llm is None:
            # vLLM is very VRAM intensive. This will likely OOM on 4GB unless 
            # it's a very small model or running with huge offloading.
            from vllm import LLM, SamplingParams
            logger.info(f"Loading vLLM model {self.model_name}...")
            self.llm = LLM(model=self.model_name, quantization="awq", enforce_eager=True)
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

    def generate(self, prompts: List[str]) -> List[str]:
        self._load_vllm()
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Simple format for vLLM generate
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant: "
        
        result = self.generate([prompt])
        return result[0].strip()

vllm_backend = vLLMBackend()
