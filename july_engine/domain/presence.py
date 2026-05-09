from __future__ import annotations
import logging
import base64
import io
import os
from PIL import Image
import numpy as np
import cv2
import inspect
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

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            res = self._strategy.get_required_vram(payload)
            if inspect.iscoroutine(res):
                return await res
            return res
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

    async def remove_background(self, payload: Dict[str, Any]):
        """
        Removes the background from an image using YOLOv11-seg.
        """
        image_data = payload.get("image")
        if not image_data:
            return None

        # Decode image
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

        from ..services.vision import character_extractor
        
        # Use CharacterExtractor to get segmented crops
        # extract_characters already handles YOLOv11-seg and GrabCut fallback
        results = character_extractor.extract_characters(image)
        
        if not results:
            return None

        # If we want to return the FULL image with background removed, 
        # we need to combine masks if multiple people are found, or just take the best one.
        # For a general "remove background" tool, combining all 'person' masks is usually best.
        
        arr = np.array(image)
        full_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        
        for res in results:
            # Re-constructing the mask for the whole image from the crop info is complex
            # so let's check if character_extractor can be improved or we do it here.
            # Actually, extract_characters returns crops. Let's do a more direct implementation here
            # similar to what I did in the standalone script for the whole image.
            pass

        # Optimized implementation for full image background removal
        if hasattr(character_extractor, 'model') and character_extractor.model is not None:
            model_results = character_extractor.model(arr, conf=0.25, verbose=False)
            found = False
            for result in model_results:
                if result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        cls = int(result.boxes.cls[i])
                        if cls == 0:  # person
                            m = mask.cpu().numpy()
                            m = cv2.resize(m, (arr.shape[1], arr.shape[0]))
                            full_mask = np.maximum(full_mask, (m * 255).astype(np.uint8))
                            found = True
            
            if not found:
                # Fallback to the first result from character_extractor if it used GrabCut
                if results and results[0]['method'] == 'grabcut_fallback':
                    # We need the full mask. Let's re-run grabcut if needed or modify extractor.
                    # For now, let's just use the crop if only one thing found
                    pass

        # Final transparency application
        rgba = cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)
        
        # If full_mask is still empty, try to get it from results or use a simple center mask
        if full_mask.max() == 0:
             # Try to reconstruct from character_extractor results
             for res in results:
                 x1, y1, x2, y2 = res['bbox']
                 # This is just a box, not a mask. 
                 # Let's just use GrabCut if YOLO failed.
                 h, w = arr.shape[:2]
                 rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
                 mask = np.zeros((h, w), np.uint8)
                 bgd = np.zeros((1, 65), np.float64)
                 fgd = np.zeros((1, 65), np.float64)
                 cv2.grabCut(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
                 full_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
                 break

        rgba[:, :, 3] = full_mask
        
        buffered = io.BytesIO()
        Image.fromarray(rgba).save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def is_loaded(self):
        return hasattr(self._strategy, "is_loaded") and self._strategy.is_loaded()

    def load(self):
        if hasattr(self._strategy, "load"):
            self._strategy.load()

    def unload(self):
        """Libera os recursos da estratégia (SD, Pix2Pix, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload()
            logger.info(f"Presence: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Presence: Strategy {self.model_tag} cleared.")
