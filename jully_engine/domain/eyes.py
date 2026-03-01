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
        if self.model_tag == "emotion":
            return Emotion(backend="cpu")
        elif self.model_tag == "fastvlm":
            return FastVLM(backend="cpu")
        elif self.backend in ["gpu", "cpu"] and self.model_tag.endswith(".gguf"):
            return GGUF(backend=self.backend)
        elif self.backend == "api":
            return LLMApi(backend=self.backend)
        else:
            # Fallback or error
            raise ValueError(f"Eyes: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def analyze(self, payload: Dict[str, Any]):
        if self.model_tag == "emotion" and self.backend == "api":
            # API implementation for emotion using a vision model (like moondream)
            image_data = payload.get("image")
            
            # If not in root, try to extract from messages
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
                                break

            if not image_data:
                raise ValueError("Eyes: No image data for emotion analysis")
            
            # Prepare a prompt that forces the model to act like the local emotion model
            valid_emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            system_prompt = f"What is the primary emotion of the person in this image? Respond with exactly one word from this list: {', '.join(valid_emotions)}."
            
            # Reconstruct payload for vision model via API
            api_payload = {
                "model": "ollama/vision_unit_tests", 
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.2,
                "headers": payload.get("headers", {})
            }
            
            # Use LLMApi to run this
            strategy = LLMApi(backend="api")
            # Extract values to avoid duplicate arguments in **api_payload
            model = api_payload.pop("model")
            messages = api_payload.pop("messages")
            
            response = strategy.run_chat(model, messages, **api_payload)
            
            # Extract content and clean it
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
            elif isinstance(response, dict):
                # litellm might return dict
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            content = content.strip().lower()
            if not content:
                logger.warning("Eyes: API vision model returned empty content")
                return "neutral" # Default fallback
                
            # Ensure it's one of the valid ones or fallback
            for em in valid_emotions:
                if em in content:
                    return em
            return content

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

        elif isinstance(self._strategy, LLMApi):
            messages = payload.get("messages", [])
            stream = payload.get("stream", False)
            base_url = payload.get("headers", {}).get("x-base-url")
            return self._strategy.run_chat(self.model_tag, messages, stream=stream, base_url=base_url)
