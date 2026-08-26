import asyncio
import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

from .adapter_base import AdapterBase

logger = logging.getLogger("JulyEngine.Adapter.ImageAdapter")

_TASK_HANDLERS = {
    "image_generation":       "_generate",
    "image_edit":             "_edit",
    "image_resize":           "_resize",
    "image_remove_background": "_remove_background",
    "video_generation":       "_generate_video",
}

# Resizer model tags handled via Pillow/OpenCV/AI resizer sub-models
_RESIZER_TAGS = frozenset(["pillow", "opencv", "gfpgan", "face_restoration", "codeformer", "realesrgan", "upscale", "lanczos", "high_quality", "onnx"])
_HEAVY_RESIZER_TAGS = frozenset(["gfpgan", "codeformer", "realesrgan", "onnx"])


class ImageAdapter(AdapterBase):
    """
    Handles all image task types: image_generation, image_edit,
    image_resize, image_remove_background, video_generation.

    Engine field: "image"

    Sub-engine selection:
      meta["alias"] == "rembg"        → BgRemoverModel
      alias in _RESIZER_TAGS          → Pillow/AI resizer sub-model
      alias == "pix2pix"              → Pix2PixModel (if available)
      alias == "lcm"                  → LCMFaceIDModel (if available)
      alias == "flux-klein"           → FluxKleinPipeline (if available)
      alias == "qwen-edit"            → QwenImageEditModel (if available)
      alias == "wan-t2v"              → Wan2T2VPipeline (if available)
      alias == "wan-i2v"              → WanI2VModel (if available)
      alias == "ltx2"                 → LTX2Model (if available)
      otherwise                       → no local strategy (callers raise)
    """

    def __init__(self, task_type: str, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(task_type, backend, model_meta)
        self._strategy = None

    @classmethod
    def get_engine_type(cls, task_type: str):
        map = {
            "image_generation":        "IMAGE_CREATE",
            "image_edit":              "IMAGE_EDIT",
            "image_resize":            "IMAGE_RESIZE",
            "image_remove_background": "IMAGE_REMOVE_BACKGROUND",
            "video_generation":        "VIDEO_GENERATION",
        }

        return map.get(task_type)

    # ------------------------------------------------------------------
    # Strategy resolution
    # ------------------------------------------------------------------

    def _alias(self) -> str:
        return self.model_id.lower()

    def _detect_engine(self) -> Optional[str]:
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

        if tag in ("qwen-edit", "qwen_edit"):
            return "qwen_edit"

        if tag in ("wan-t2v", "wan2-t2v", "wan_t2v"):
            return "wan_t2v"

        if tag in ("wan-i2v", "wan2-i2v", "wan_i2v"):
            return "wan_i2v"

        if tag in ("ltx2", "ltx-2", "ltx_2"):
            return "ltx2"

        return None

    def _get_strategy(self):
        if self._strategy is not None:
            return self._strategy

        engine = self._detect_engine()
        logger.debug(f"ImageAdapter: detected engine='{engine}' for tag='{self._alias()}'")

        if engine == "rembg":
            from ..models.bg_remover import BgRemoverModel
            self._strategy = BgRemoverModel(backend=self.backend, model_meta=self.meta)

        elif engine and engine.startswith("resize_"):
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
                self._strategy = FluxKleinPipeline(backend=self.backend, model_meta=self.meta)
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

        elif engine == "qwen_edit":
            try:
                from ..models.qwen_image_edit import QwenImageEditModel
                self._strategy = QwenImageEditModel(backend=self.backend, model_meta=self.meta)
            except Exception as e:
                logger.warning(f"ImageAdapter: qwen-edit model could not be loaded: {str(e)}")
                self._strategy = None

        elif engine == "wan_t2v":
            try:
                from ..models.wan2_t2v import Wan2T2VPipeline
                self._strategy = Wan2T2VPipeline(backend=self.backend, model_meta=self.meta)
            except Exception as e:
                logger.warning(f"ImageAdapter: wan-t2v model could not be loaded: {str(e)}")
                self._strategy = None

        elif engine == "wan_i2v":
            try:
                from ..models.wan2_i2v import WanI2VModel
                self._strategy = WanI2VModel(backend=self.backend, model_meta=self.meta)
            except Exception as e:
                logger.warning(f"ImageAdapter: wan-i2v model could not be loaded: {str(e)}")
                self._strategy = None

        elif engine == "ltx2":
            try:
                from ..models.ltx2_video import LTX2Model
                self._strategy = LTX2Model(backend=self.backend, model_meta=self.meta)
            except Exception as e:
                logger.warning(f"ImageAdapter: ltx2 model could not be loaded: {str(e)}")
                self._strategy = None

        return self._strategy

    def _make_resizer(self, tag: str):
        try:
            from ..models.image_resizer import (
                PillowResizerModel, OpencvResizerModel,
                GFPGANResizerModel, CodeFormerResizerModel, RealESRGANResizerModel,
                LanczosResizerModel, HighQualityUpscalerModel, OnnxUpscalerModel
            )
            resizer_map = {
                "pillow":          PillowResizerModel,
                "opencv":          OpencvResizerModel,
                "gfpgan":          GFPGANResizerModel,
                "face_restoration": GFPGANResizerModel,
                "codeformer":      CodeFormerResizerModel,
                "realesrgan":      RealESRGANResizerModel,
                "upscale":         RealESRGANResizerModel,
                "lanczos":         LanczosResizerModel,
                "high_quality":    HighQualityUpscalerModel,
                "onnx":            OnnxUpscalerModel,
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
        strategy = self._get_strategy()
        if strategy is None:
            raise ValueError(f"ImageAdapter: no local image-generation model available for tag '{self._alias()}'")

        tag = self._alias()

        if tag == "pix2pix":
            # Mesmo bug de assinatura do branch em `_edit` acima.
            img = self._blank_image(payload.get("width", 512), payload.get("height", 512))
            return strategy.run(
                img,
                payload.get("prompt"),
                width=payload.get("width"),
                height=payload.get("height"),
            )

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
        strategy = self._get_strategy()
        if strategy is None:
            raise ValueError(f"ImageAdapter: no local image-edit model available for tag '{self._alias()}'")

        tag = self._alias()

        if tag == "pix2pix":
            # Achado real: `Pix2PixPipeline.run(self, image_data: str, prompt:
            # str, **kwargs)` recebia um dict inteiro como único argumento
            # posicional (nunca o `prompt` de verdade) -- TypeError garantido,
            # sempre virava 500 genérico na API. `width`/`height` também
            # nunca chegavam, então nenhum controle de resolução funcionava.
            return strategy.run(
                payload.get("image"),
                payload.get("prompt"),
                width=payload.get("width"),
                height=payload.get("height"),
            )

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

        if tag in ("qwen-edit", "qwen_edit"):
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

    async def _generate_video(self, payload: Dict[str, Any]):
        strategy = self._get_strategy()
        if strategy is None:
            raise ValueError(f"ImageAdapter: no local video-generation model available for tag '{self._alias()}'")

        # strategy.run() is an async generator for every video-producing model —
        # returning it here (without awaiting) lets the orchestrator's Runner detect
        # __aiter__ and stream the result back in chunks instead of buffering the
        # whole clip in memory.
        return strategy.run(payload)

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
    def _blank_image(width: int = 512, height: int = 512) -> str:
        from PIL import Image
        img = Image.new("RGB", (width, height), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
