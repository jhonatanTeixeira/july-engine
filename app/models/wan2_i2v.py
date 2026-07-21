import gc
import io
import base64
import asyncio
import logging
import tempfile
from typing import Optional, Dict, Any
from PIL import Image

try:
    from .sdnq_diffusion_base import SDNQDiffusionModel
except ImportError:
    from sdnq_diffusion_base import SDNQDiffusionModel

logger = logging.getLogger("JulyEngine.Models.WanI2V")

_VRAM_TIERS = {"sequential": 2000, "cpu": 4000, "none": 10000}


class WanI2VModel(SDNQDiffusionModel):
    DEFAULT_MODEL_ID = "Disty0/Wan2.2-I2V-A14B-SDNQ-uint4-svd-r32"
    OFFLOAD_ENV_VAR = "WAN_I2V_OFFLOAD"
    VRAM_TIERS = _VRAM_TIERS

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if self.is_loaded():
            return

        logger.info(f"WanI2V: Inicializando carga de {self.model_id}...")

        import torch
        from diffusers import WanImageToVideoPipeline
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )

        if torch.cuda.is_available() and triton_is_available:
            logger.info("WanI2V: Triton detectado. Aplicando matmul otimizado ao transformer...")
            self.pipeline.transformer = apply_sdnq_options_to_model(
                self.pipeline.transformer, use_quantized_matmul=True
            )
            try:
                self.pipeline.text_encoder = apply_sdnq_options_to_model(
                    self.pipeline.text_encoder, use_quantized_matmul=True
                )
            except Exception:
                pass
        else:
            logger.info("WanI2V: Triton ausente. Usando Eager Mode do PyTorch.")

        self._apply_offload(self.pipeline)
        logger.info("WanI2V: Carga finalizada com sucesso!")

    @staticmethod
    def _decode_image(image_data) -> "Image.Image":
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def _render(self, payload: Dict[str, Any]) -> str:
        """Blocking render + export to a temp mp4 file — runs inside asyncio.to_thread."""
        import torch
        from diffusers.utils import export_to_video

        image_data = payload.get("image")
        if not image_data:
            raise ValueError("WanI2V: 'image' field is required for image-to-video generation")
        image = self._decode_image(image_data)

        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "") or None
        height = int(payload.get("height", 480))
        width = int(payload.get("width", 832))
        num_frames = int(payload.get("num_frames", 81))
        num_inference_steps = int(payload.get("num_inference_steps", 40))
        guidance_scale = float(payload.get("guidance_scale", 3.5))
        fps = int(payload.get("fps", 16))
        seed = int(payload.get("seed", -1))

        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"WanI2V: Gerando vídeo {width}x{height} com {num_frames} frames, {num_inference_steps} steps...")

        output = self.pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        frames = output.frames[0]
        logger.info(f"WanI2V: {len(frames)} frames gerados. Exportando para MP4...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=fps)
        return tmp_path

    async def run(self, payload: Dict[str, Any], **kwargs):
        if not self.is_loaded():
            await asyncio.to_thread(self.load)

        async with self._inference_lock:
            try:
                tmp_path = await asyncio.to_thread(self._render, payload)
            except Exception as e:
                logger.error(f"WanI2V: Erro fatal na inferência: {e}")
                raise

            async for chunk in self._stream_file(tmp_path):
                yield chunk
