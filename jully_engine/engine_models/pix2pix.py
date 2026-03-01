import os
import logging
import torch
from typing import Any, Dict
from PIL import Image
import io
import base64

logger = logging.getLogger("JulyEngine.Models.Pix2Pix")

class Pix2Pix:
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cuda" if backend == "gpu" and torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.model_id = "timbrooks/instruct-pix2pix"

    def load(self):
        if self.pipeline is None:
            try:
                from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
                logger.info(f"Pix2Pix: Loading model {self.model_id} on {self.device}")
                self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
                    safety_checker=None
                ).to(self.device)
                self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
                logger.info("Pix2Pix loaded successfully.")
            except Exception as e:
                logger.error(f"Pix2Pix: Failed to load: {e}")
                raise e

    def run(self, image_data: str, prompt: str) -> str:
        if self.pipeline is None:
            self.load()
            
        try:
            # Decode image
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Run inference
            edited_image = self.pipeline(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images[0]

            # Encode back to base64
            buffered = io.BytesIO()
            edited_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Pix2Pix: Execution failed: {e}")
            raise e
