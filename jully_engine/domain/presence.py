from __future__ import annotations
import logging
import base64
import io
import os
from PIL import Image
from typing import Any, Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_models.pix2pix import Pix2Pix
    from ..engine_models.llm_api import LLMApi
    from ..engine_models.stable_diffusion_lcm import LCMFaceIDPipeline
    from ..engine_models.stable_diffusion_video import LCMVideoPipeline
    from ..engine_models.flux_klein import FluxKleinNode

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
        tag = self.model_tag.lower() if self.model_tag else ""
        
        if self.backend == "api":
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)
        elif tag == "pix2pix":
            from ..engine_models.pix2pix import Pix2Pix
            return Pix2Pix(backend=self.backend)
        elif tag == "lcm":
            from ..engine_models.stable_diffusion_lcm import LCMFaceIDPipeline
            return LCMFaceIDPipeline(use_face_id=False, use_cpu_offload=True)
        elif tag == "video":
            from ..engine_models.stable_diffusion_video import LCMVideoPipeline
            return LCMVideoPipeline()
        elif tag == "flux-klein":
            from ..engine_models.flux_klein import FluxKleinNode
            return FluxKleinNode(backend=self.backend)
        elif tag == "pillow":
            from ..engine_models.resize import PillowResizer
            return PillowResizer()
        elif tag == "opencv":
            from ..engine_models.resize import OpencvResizer
            return OpencvResizer()
        elif tag in ["gfpgan", "face_restoration"]:
            from ..engine_models.resize import GFPGANResizer
            return GFPGANResizer()
        elif tag == "codeformer":
            from ..engine_models.resize import CodeFormerResizer
            return CodeFormerResizer()
        elif tag in ["realesrgan", "upscale"]:
            from ..engine_models.resize import RealESRGANResizer
            return RealESRGANResizer()
        else:
            # Qualquer outra coisa é considerada LLM/API
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            return self._strategy.get_required_vram(payload)
        return 0

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
        from ..engine_models.pix2pix import Pix2Pix
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.stable_diffusion_lcm import LCMFaceIDPipeline
        from ..engine_models.flux_klein import FluxKleinNode
        from ..persistence import get_backend

        headers: dict = payload.setdefault("headers", {})
        config: dict = get_backend().get_setting("IMAGE_EDIT")

        if config:
            headers.setdefault("x-base-url", config.get("base_url"))
            headers.setdefault("x-api-key", config.get("api_key"))
                
        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            image_data = payload.pop("image", "")
            mask_data = payload.pop("mask", None)
            prompt = payload.pop("prompt", "")
            headers = payload.pop("headers", headers)
            
            # Helper interno limpo para não duplicar código
            def decode_b64_to_file(b64_str: str, filename="image.png"):
                if isinstance(b64_str, str) and b64_str.startswith("data:image"):
                    b64_str = b64_str.split(",")[1]
                
                img_bytes = base64.b64decode(b64_str)
                img_file = io.BytesIO(img_bytes)
                img_file.name = filename
                
                return img_file

            img_file = decode_b64_to_file(image_data, "image.png")
            mask_file = decode_b64_to_file(mask_data, "mask.png") if mask_data else None
            
            return await self._strategy.run_image_edit(
                model=model, 
                prompt=prompt, 
                image=img_file, 
                mask=mask_file,
                headers=headers, 
                **payload
            )
        
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
                guidance_scale=payload.get("guidance_scale", 1.5),
                width=payload.get("width", 512),
                height=payload.get("height", 512)
            )
            
            if images:
                buffered = io.BytesIO()
                images[0].save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            
        elif isinstance(self._strategy, FluxKleinNode):
            return self._strategy.run(payload)
            
        return None

    async def resize(self, payload: Dict[str, Any]):
        """
        Redimensiona ou faz upscale de uma imagem.
        Pode usar redimensionamento clássico (Pillow/OpenCV) ou modelos de IA.
        Se o backend for 'image_edit_model', roteia para o método edit (Pix2Pix/API).
        """
        try:
            if self.backend == "image_edit_model" or payload.get("is_image_edit_route"):
                payload["prompt"] = payload.get("prompt", "upscale this image to high quality, clear details, 4k")
                return await self.edit(payload)
            
            # Se a estratégia tiver o método resize (nossos resizers manuais)
            if hasattr(self._strategy, "resize"):
                result = self._strategy.resize(payload)
                # Modelos de IA pesados devem liberar VRAM após um resize (one-shot task)
                if self.model_tag.lower() in ["gfpgan", "codeformer", "realesrgan"]:
                    self.unload()
                return result
                
            # Se for qualquer outro modelo (LLM/API), usamos o edit como fallback de IA
            payload["prompt"] = payload.get("prompt", "upscale this image to high quality, clear details, 4k")
            return await self.edit(payload)
        except Exception as e:
            logger.error(f"Presence: Resize failed: {e}")
            self.unload()
            raise e

    async def generate(self, payload: Dict[str, Any]):
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.pix2pix import Pix2Pix
        from ..engine_models.stable_diffusion_lcm import LCMFaceIDPipeline
        from ..engine_models.flux_klein import FluxKleinNode

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
            img = Image.new('RGB', (payload.get("width", 512), payload.get("height", 512)), color = 'white')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode()
            return self._strategy.run(
                image_data, 
                payload.get("prompt"), 
                width=payload.get("width"), 
                height=payload.get("height")
            )
            
        elif isinstance(self._strategy, LCMFaceIDPipeline):
            images = self._strategy(
                prompt=payload.get("prompt", ""),
                num_inference_steps=payload.get("steps", 6),
                guidance_scale=payload.get("guidance_scale", 1.5),
                width=payload.get("width", 512),
                height=payload.get("height", 512)
            )
            if images:
                buffered = io.BytesIO()
                images[0].save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()

        elif isinstance(self._strategy, FluxKleinNode):
            return self._strategy.run(payload)
            
        return None

    async def generate_video(self, payload: Dict[str, Any]):
        from ..engine_models.stable_diffusion_video import LCMVideoPipeline
        
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
