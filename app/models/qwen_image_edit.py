import os
import gc
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image

try:
    from .sdnq_diffusion_base import SDNQDiffusionModel
except ImportError:
    from sdnq_diffusion_base import SDNQDiffusionModel

logger = logging.getLogger("JulyEngine.Models.QwenImageEdit")


class QwenImageEditModel(SDNQDiffusionModel):
    DEFAULT_MODEL_ID = "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32"
    OFFLOAD_ENV_VAR = "QWEN_EDIT_OFFLOAD"
    VRAM_TIERS = {"sequential": 1500, "cpu": 2500, "none": 6000}

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if self.is_loaded():
            return

        logger.info(f"QwenImageEdit: Inicializando carga de {self.model_id}...")

        import torch
        from diffusers import QwenImageEditPlusPipeline
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )

        if torch.cuda.is_available() and triton_is_available:
            logger.info("QwenImageEdit: Triton detectado. Aplicando matmul otimizado...")
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
            logger.info("QwenImageEdit: Triton ausente. Usando Eager Mode do PyTorch.")

        self._apply_offload(self.pipeline)
        logger.info("QwenImageEdit: Carga finalizada com sucesso!")

    def run(self, payload: Dict[str, Any], **kwargs):
        if not self.is_loaded():
            self.load()

        import torch

        image_data = payload.get("image")
        if not image_data:
            raise ValueError("QwenImageEdit: 'image' field is required for editing")

        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            image = Image.open(BytesIO(image_data)).convert("RGB")

        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "") or None
        num_inference_steps = int(payload.get("num_inference_steps", 40))
        guidance_scale = float(payload.get("guidance_scale", 1.0))
        true_cfg_scale = float(payload.get("true_cfg_scale", 4.0))
        num_images_per_prompt = int(payload.get("num_images_per_prompt", 1))
        seed = int(payload.get("seed", -1))

        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        try:
            result = self.pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )
            result_image = result.images[0]

            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"QwenImageEdit: Erro fatal na inferência: {e}")
            raise
