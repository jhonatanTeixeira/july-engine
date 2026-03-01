import logging
from typing import Any, Dict, Optional
from ..engine_models.pix2pix import Pix2Pix
from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Presence")

class Presence:
    """
    Handles image editing and generation.
    Strategies: Pix2Pix (gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.model_tag == "pix2pix":
            return Pix2Pix(backend=self.backend)
        elif self.backend == "api":
            return LLMApi(backend=self.backend)
        else:
            raise ValueError(f"Presence: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def edit(self, payload: Dict[str, Any]):
        if isinstance(self._strategy, Pix2Pix):
            image_data = payload.get("image")
            prompt = payload.get("prompt")
            return self._strategy.run(image_data, prompt)
        elif isinstance(self._strategy, LLMApi):
            # litellm image edit
            image_data = payload.get("image")
            prompt = payload.get("prompt")
            base_url = payload.get("headers", {}).get("x-base-url")
            
            # Resolve base64 string to bytes if needed for litellm.image_editing
            import base64
            import io
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            
            img_bytes = base64.b64decode(image_data)
            img_file = io.BytesIO(img_bytes)
            img_file.name = "image.png"
            return self._strategy.run_image_edit(self.model_tag, prompt, img_file, base_url=base_url)
        return None

    async def generate(self, payload: Dict[str, Any]):
        if isinstance(self._strategy, Pix2Pix):
            # Fallback to edit with a blank image for generation
            from PIL import Image
            import io
            import base64
            img = Image.new('RGB', (512, 512), color = 'white')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode()
            return self._strategy.run(image_data, payload.get("prompt"))
        elif isinstance(self._strategy, LLMApi):
            prompt = payload.get("prompt")
            base_url = payload.get("headers", {}).get("x-base-url")
            return self._strategy.run_image_gen(self.model_tag, prompt, base_url=base_url)
        return None
