import logging
import base64
import io
from PIL import Image
from typing import Any, Dict, List, Optional
from ..engine_models.gguf import GGUF
from ..engine_models.fastvlm import FastVLM
from ..engine_models.emotion import Emotion
from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Eyes")

class Eyes:
    """
    Handles vision and image analysis.
    Strategies: GGUF (cpu, gpu), FastVLM (cpu), Emotion (cpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        elif self.model_tag == "emotion":
            return Emotion(backend="cpu")
        elif self.model_tag == "fastvlm":
            return FastVLM(backend="cpu")
        
        from ..routers.models import load_models_db
        db = load_models_db()
        is_gguf = self.model_tag.endswith(".gguf") or self.model_tag in db

        if self.backend in ["gpu", "cpu"] and is_gguf:
            return GGUF(backend=self.backend, model_alias=self.model_tag)
        else:
            raise ValueError(f"Eyes: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def analyze(self, payload: Dict[str, Any]):
        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            headers = payload.pop("headers", {})
            return await self._strategy.run_chat(model, messages, stream=stream, headers=headers, **payload)

        # Generalize extraction for local models
        image_data = payload.get("image")
        prompt = payload.get("prompt", "")
        
        if not image_data:
            messages = payload.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                image_data = url.split(",")[1]
                            else:
                                image_data = url
                        elif part.get("type") == "text":
                            prompt = part.get("text", prompt)
            
            if image_data:
                payload["image"] = image_data
            if prompt:
                payload["prompt"] = prompt

        if isinstance(self._strategy, Emotion):
            image_data = payload.get("image")
            if not image_data:
                raise ValueError("Eyes: No image data for emotion analysis")
            
            # Decode if needed
            if isinstance(image_data, str):
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",")[1]
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                
            return self._strategy.run(img)

        elif isinstance(self._strategy, FastVLM):
            return self._strategy.run(payload)

        elif isinstance(self._strategy, GGUF):
            messages = payload.get("messages", [])
            stream = payload.get("stream", False)
            return self._strategy.run_chat(self.model_tag, messages, stream=stream)
