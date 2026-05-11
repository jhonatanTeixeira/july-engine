import asyncio
import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

from ..models.base_model import BaseModel

logger = logging.getLogger("JulyEngine.Adapter.ImageAdapter")

_TASK_HANDLERS = {
    "image_generation":       "_generate",
    "image_edit":             "_edit",
    "image_resize":           "_resize",
    "image_remove_background": "_remove_background",
}

# Resizer model tags handled via Pillow/OpenCV/AI resizer sub-models
_RESIZER_TAGS = frozenset(["pillow", "opencv", "gfpgan", "face_restoration", "codeformer", "realesrgan", "upscale"])
_HEAVY_RESIZER_TAGS = frozenset(["gfpgan", "codeformer", "realesrgan"])


class ImageAdapter(BaseModel):
    """
    Handles all image task types: image_generation, image_edit,
    image_resize, image_remove_background.

    Engine field: "image"

    Sub-engine selection:
      backend == "api"          → LLMApi for gen/edit
      meta["alias"] == "rembg"  → BgRemoverModel
      alias in _RESIZER_TAGS    → Pillow/AI resizer sub-model
      alias == "pix2pix"        → Pix2PixModel (if available)
      alias == "lcm"            → LCMFaceIDModel (if available)
      alias == "flux-klein"     → FluxKleinModel (if available)
      otherwise (api)           → LLMApi fallback
    """

    def __init__(self, task_type: str, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._strategy = None
        self.task_type = task_type

    # ------------------------------------------------------------------
    # Strategy resolution
    # ------------------------------------------------------------------

    def _alias(self) -> str:
        return self.model_id.lower()

    def _detect_engine(self) -> str:
        if self.backend == "api":
            return "api"
        
        tag = self._alias()
        
        if tag == "rembg":
            return "rembg"
        
        if tag in _RESIZER_TAGS:
            return f"resize_{tag}"
        
        if tag == "pix2pix":
            return "pix2pix"
        
        if tag == "lcm":
            return "lcm"
        
        if tag in ("flux-klein", "flux_klein"):
            return "flux"
        
        if tag == "video":
            return "video"
        
        return "api"

    def _get_strategy(self):
        if self._strategy is not None:
            return self._strategy

        engine = self._detect_engine()
        logger.debug(f"ImageAdapter: detected engine='{engine}' for tag='{self._alias()}'")

        if engine == "rembg":
            from ..models.bg_remover import BgRemoverModel
            self._strategy = BgRemoverModel(backend=self.backend, model_meta=self.meta)

        elif engine.startswith("resize_"):
            tag = engine.removeprefix("resize_")
            self._strategy = self._make_resizer(tag)

        elif engine == "pix2pix":
            try:
                from ..models.pix2pix import Pix2PixPipeline
                self._strategy = Pix2PixPipeline(device="cuda" if self.backend == "gpu" else "cpu", use_sequential_offload=True)
            except Exception as e:
                logger.warning(f"ImageAdapter: pix2pix model could not be loaded: {str(e)}")
                self._strategy = None

        elif engine == "lcm":
            try:
                from ..models.stable_diffusion_lcm import LCMFaceIDPipeline
                self._strategy = LCMFaceIDPipeline(device="cuda" if self.backend == "gpu" else "cpu", use_sequential_offload=True)
            except Exception as e:
                logger.warning(f"ImageAdapter: lcm model could not be loaded: {str(e)}")
                self._strategy = None

        elif engine == "flux":
            try:
                from ..models.flux_klein import FluxKleinPipeline
                self._strategy = FluxKleinPipeline(device="cuda" if self.backend == "gpu" else "cpu", use_sequential_offload=True)
            except Exception as e:
                logger.warning(f"ImageAdapter: flux-klein model could not be loaded: {str(e)}")
                self._strategy = None

        elif engine == "video":
            try:
                from ..models.stable_diffusion_video import LCMVideoPipeline
                self._strategy = LCMVideoPipeline(device="cuda" if self.backend == "gpu" else "cpu", use_sequential_offload=True)
            except Exception as e:
                logger.warning(f"ImageAdapter: video model could not be loaded: {str(e)}")
                self._strategy = None

        return self._strategy

    def _make_resizer(self, tag: str):
        try:
            from ..models.image_resizer import (
                PillowResizerModel, OpencvResizerModel,
                GFPGANResizerModel, CodeFormerResizerModel, RealESRGANResizerModel,
            )
            resizer_map = {
                "pillow":          PillowResizerModel,
                "opencv":          OpencvResizerModel,
                "gfpgan":          GFPGANResizerModel,
                "face_restoration": GFPGANResizerModel,
                "codeformer":      CodeFormerResizerModel,
                "realesrgan":      RealESRGANResizerModel,
                "upscale":         RealESRGANResizerModel,
            }
            cls = resizer_map.get(tag)
            if cls:
                return cls(backend=self.backend, model_meta=self.meta)
        except ImportError:
            logger.warning(f"ImageAdapter: resizer model '{tag}' not available, using Pillow fallback")
            try:
                from ..models.image_resizer import PillowResizerModel
                return PillowResizerModel(backend=self.backend, model_meta=self.meta)
            except ImportError:
                pass
        return None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        s = self._get_strategy()
        if s and hasattr(s, "get_required_vram"):
            result = s.get_required_vram(payload)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return 0

    def load(self, n_ctx=None, num_layers=None):
        s = self._get_strategy()
        if s and hasattr(s, "load"):
            s.load()

    def is_loaded(self) -> bool:
        return self._strategy is not None and self._strategy.is_loaded()

    def unload(self, model_name=None):
        if self._strategy and hasattr(self._strategy, "unload"):
            self._strategy.unload()

    async def run(self, payload: Dict[str, Any]):
        task_type = self.task_type
        handler_name = _TASK_HANDLERS.get(task_type)
        
        if not handler_name:
            raise ValueError(f"ImageAdapter: unknown task_type '{task_type}'")
        return await getattr(self, handler_name)(payload)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _generate(self, payload: Dict[str, Any]):
        engine = self._detect_engine()
        headers = payload.get("headers", {})

        config = self._load_config("IMAGE_CREATE")
        if config:
            headers.setdefault("x-base-url", config.get("base_url"))
            if config.get("api_key"):
                headers.setdefault("x-api-key", config["api_key"])
                headers.setdefault("authorization", f"Bearer {config['api_key']}")

        if engine in ("api", None) or self._get_strategy() is None:
            from ..services.llm_api import llm_api
            return await llm_api.dispatch("image_generation", {**payload, "headers": headers})

        strategy = self._get_strategy()
        tag = self._alias()

        if tag == "pix2pix":
            img = self._blank_image(payload.get("width", 512), payload.get("height", 512))
            return strategy.run({"image": img, **payload})

        if tag == "lcm":
            images = strategy(
                prompt=payload.get("prompt", ""),
                num_inference_steps=payload.get("steps", 6),
                guidance_scale=payload.get("guidance_scale", 1.5),
                width=payload.get("width", 512),
                height=payload.get("height", 512),
            )
            if images:
                return self._pil_to_b64(images[0])

        if tag in ("flux-klein", "flux_klein"):
            return strategy.run(payload)

        return None

    async def _edit(self, payload: Dict[str, Any]):
        engine = self._detect_engine()
        headers = payload.get("headers", {})

        config = self._load_config("IMAGE_EDIT")
        if config:
            headers.setdefault("x-base-url", config.get("base_url"))
            if config.get("api_key"):
                headers.setdefault("x-api-key", config["api_key"])

        if engine in ("api", None) or self._get_strategy() is None:
            from ..services.llm_api import llm_api
            image_data = payload.get("image", "")
            mask_data = payload.get("mask")
            return await llm_api.dispatch("image_edit", {
                **payload,
                "image": self._b64_to_file(image_data, "image.png"),
                "mask": self._b64_to_file(mask_data, "mask.png") if mask_data else None,
                "headers": headers,
            })

        strategy = self._get_strategy()
        tag = self._alias()

        if tag == "pix2pix":
            return strategy.run({"image": payload.get("image"), "prompt": payload.get("prompt")})

        if tag == "lcm":
            face_img = self._decode_pil(payload.get("image"))
            images = strategy(
                prompt=payload.get("prompt", ""),
                face_image=face_img,
                num_inference_steps=payload.get("steps", 10),
                guidance_scale=payload.get("guidance_scale", 1.5),
                width=payload.get("width", 512),
                height=payload.get("height", 512),
            )
            if images:
                return self._pil_to_b64(images[0])

        if tag in ("flux-klein", "flux_klein"):
            return strategy.run(payload)

        return None

    async def _resize(self, payload: Dict[str, Any]):
        # 3-level fallback: edit route → resizer method → edit fallback
        if self.backend == "image_edit_model" or payload.get("is_image_edit_route"):
            payload.setdefault("prompt", "upscale this image to high quality, clear details, 4k")
            return await self._edit(payload)

        strategy = self._get_strategy()
        if strategy and hasattr(strategy, "resize"):
            result = strategy.resize(payload)
            tag = self._alias()
            if tag in _HEAVY_RESIZER_TAGS:
                self.unload()
            return result

        # fallback to edit with upscale prompt
        payload.setdefault("prompt", "upscale this image to high quality, clear details, 4k")
        return await self._edit(payload)

    async def _remove_background(self, payload: Dict[str, Any]) -> str:
        strategy = self._get_strategy()
        if strategy is None:
            raise ValueError("ImageAdapter: no rembg strategy loaded for image_remove_background")

        image_data = payload.get("image")
        if not image_data:
            raise ValueError("ImageAdapter: 'image' field required for remove_background")

        pil_img = self._decode_pil(image_data)
        output_img = strategy.run({"image": pil_img})

        if hasattr(output_img, "save"):
            return self._pil_to_b64(output_img)
        # already base64 or bytes
        if isinstance(output_img, bytes):
            return base64.b64encode(output_img).decode()
        return output_img

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_pil(image_data):
        from PIL import Image
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        return image_data

    @staticmethod
    def _pil_to_b64(img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _b64_to_file(b64_str: str, filename: str):
        if not b64_str:
            return None
        if b64_str.startswith("data:image"):
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        buf = io.BytesIO(img_bytes)
        buf.name = filename
        return buf

    @staticmethod
    def _blank_image(width: int = 512, height: int = 512) -> str:
        from PIL import Image
        img = Image.new("RGB", (width, height), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _load_config(self, key: str) -> dict:
        try:
            from ..services.models_service import model_service
            return model_service.backend.get_setting(key) or {}
        except Exception:
            return {}
