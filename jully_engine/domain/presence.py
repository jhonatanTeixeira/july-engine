import logging
import base64
import io
import os
from PIL import Image
from typing import Any, Dict, Optional, List

from ..engine_models.pix2pix import Pix2Pix
from ..engine_models.llm_api import LLMApi
from ..engine_models.stable_diffusion_lcm import LCMFaceIDPipeline
from ..engine_models.stable_diffusion_video import LCMVideoPipeline

logger = logging.getLogger("JulyEngine.Domain.Presence")

class Presence:
    """
    Handles image editing and generation.
    Strategies: Pix2Pix (gpu), LLMApi (api), LCMFaceIDPipeline (gpu), LCMVideoPipeline (gpu).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        elif self.model_tag == "pix2pix":
            return Pix2Pix(backend=self.backend)
        elif self.model_tag == "lcm":
            return LCMFaceIDPipeline(use_face_id=False, use_cpu_offload=True)
        elif self.model_tag == "video":
            return LCMVideoPipeline()
        else:
            raise ValueError(f"Presence: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def _find_last_image(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Busca a última imagem enviada nas mensagens, do fim para o começo."""
        for msg in reversed(messages):
            content = msg.get("content")
            if isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return part["image_url"]["url"]
            elif isinstance(content, str) and content.startswith("data:image"):
                return content
        return None

    async def edit(self, payload: Dict[str, Any]):
        headers = payload.get("headers", {})
        
        from ..persistence import get_backend
        config = get_backend().get_setting("IMAGE_EDIT")
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"
                
        image_data = payload.get("image")
        messages = payload.get("messages", [])
        
        # Se não tem imagem no payload, tenta buscar no conteúdo multimodal ou no histórico
        if not image_data and messages:
            image_data = self._find_last_image(messages)
            if image_data:
                payload["image"] = image_data

        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            image_data = payload.pop("image", "")
            prompt = payload.pop("prompt", "")
            headers = payload.pop("headers", headers)
            
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            
            img_bytes = base64.b64decode(image_data)
            img_file = io.BytesIO(img_bytes)
            img_file.name = "image.png"
            return await self._strategy.run_image_edit(model, prompt, img_file, headers=headers, **payload)
            
        elif isinstance(self._strategy, Pix2Pix):
            image_data = payload.get("image")
            prompt = payload.get("prompt")
            return self._strategy.run(image_data, prompt)
            
        elif isinstance(self._strategy, LCMFaceIDPipeline):
            prompt = payload.get("prompt", "")
            image_data = payload.get("image")
            
            face_img = None
            if image_data:
                if isinstance(image_data, str):
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(",")[1]
                    img_bytes = base64.b64decode(image_data)
                    face_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    face_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            images = self._strategy(
                prompt=prompt,
                face_image=face_img,
                num_inference_steps=payload.get("steps", 10),
                guidance_scale=payload.get("guidance_scale", 1.5)
            )
            
            if images:
                buffered = io.BytesIO()
                images[0].save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            
        return None

    async def generate(self, payload: Dict[str, Any]):
        headers = payload.get("headers", {})
        
        from ..persistence import get_backend
        config = get_backend().get_setting("IMAGE_CREATE")
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"

        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            prompt = payload.pop("prompt", "")
            headers = payload.pop("headers", headers)
            return await self._strategy.run_image_gen(model, prompt, headers=headers, **payload)
            
        elif isinstance(self._strategy, Pix2Pix):
            img = Image.new('RGB', (512, 512), color = 'white')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode()
            return self._strategy.run(image_data, payload.get("prompt"))
            
        elif isinstance(self._strategy, LCMFaceIDPipeline):
            images = self._strategy(
                prompt=payload.get("prompt", ""),
                num_inference_steps=payload.get("steps", 6),
                guidance_scale=payload.get("guidance_scale", 1.5)
            )
            if images:
                buffered = io.BytesIO()
                images[0].save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()

        return None

    async def generate_video(self, payload: Dict[str, Any]):
        if isinstance(self._strategy, LCMVideoPipeline):
            frames = self._strategy.generate_video(
                prompt=payload.get("prompt", ""),
                negative_prompt=payload.get("negative_prompt", ""),
                num_frames=payload.get("num_frames", 16),
                width=payload.get("width", 384),
                height=payload.get("height", 384),
                steps=payload.get("steps", 6)
            )
            
            from diffusers.utils import export_to_gif
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                export_to_gif(frames, tmp.name)
                with open(tmp.name, "rb") as f:
                    gif_data = f.read()
                os.unlink(tmp.name)
                return base64.b64encode(gif_data).decode()
        
        return None

    def unload(self):
        """Libera os recursos da estratégia (SD, Pix2Pix, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload()
            logger.info(f"Presence: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Presence: Strategy {self.model_tag} cleared.")
