import gc
import logging
from typing import Optional, Dict, Any

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.Molmo")

MODEL_ID = "CalamitousFelicitousness/Molmo-7B-O-0924-SDNQ-UINT4-SVD-R32"


class MolmoModel(BaseModel):
    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.model_id = self.meta.get("id", MODEL_ID)
        self._model = None
        self._processor = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 6000

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device_map = "auto" if (self.backend == "gpu" and torch.cuda.is_available()) else "cpu"

        logger.info(f"Molmo: Carregando {self.model_id} (device_map={device_map})...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
        )

        # Best-effort post-load SDNQ optimization — no established precedent in this
        # codebase for SDNQ + transformers.AutoModelForCausalLM (only diffusers
        # pipelines use this today), so failures here are non-fatal.
        try:
            from sdnq.loader import apply_sdnq_options_to_model
            self._model = apply_sdnq_options_to_model(self._model, use_quantized_matmul=True)
            logger.info("Molmo: SDNQ matmul otimizado aplicado.")
        except Exception as e:
            logger.warning(f"Molmo: SDNQ optimization failed ({e}), continuando sem ele.")

        logger.info("Molmo: Carga finalizada com sucesso!")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name: Optional[str] = None):
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, payload: Dict[str, Any], **kwargs):
        if self._model is None:
            self.load()

        from transformers import GenerationConfig

        image_data = payload.get("image")
        prompt = payload.get("prompt") or "Describe this image."

        if not image_data:
            return ""

        image = self._decode(image_data)

        inputs = self._processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self._model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self._processor.tokenizer,
        )

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text.strip()

    @staticmethod
    def _decode(image_data):
        from PIL import Image
        import base64, io, re

        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]
            elif image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[-1]
            image_data = re.sub(r"[^a-zA-Z0-9+/=]", "", image_data)
            pad = len(image_data) % 4
            if pad:
                image_data += "=" * (4 - pad)
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return Image.open(io.BytesIO(image_data)).convert("RGB")
