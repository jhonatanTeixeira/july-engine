import logging
from typing import Any, Dict, Optional
from PIL import Image
import io
import base64
import os

logger = logging.getLogger("JulyEngine.Models.BgRemover")

class BgRemover:
    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self.session = None
        
    def load(self):
        if self.session is not None:
            return
            
        try:
            from rembg import new_session
            import onnxruntime as ort
            
            # Determine execution providers based on availability
            available = ort.get_available_providers()
            providers = []
            
            if self.backend == "gpu":
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                elif "VulkanExecutionProvider" in available:
                    providers.append("VulkanExecutionProvider")
                elif "ROCMExecutionProvider" in available:
                    providers.append("ROCMExecutionProvider")
                
                logger.info(f"BgRemover: Attempting GPU providers: {providers}")
            
            providers.append("CPUExecutionProvider")
                
            # U-2-Net is the default and robust model
            self.session = new_session(model_name="u2net", providers=providers)
            logger.info(f"BgRemover: Loaded with providers {providers}")
        except Exception as e:
            logger.error(f"BgRemover: Failed to load: {e}")
            raise e

    def run(self, image: Image.Image) -> Image.Image:
        from rembg import remove
        
        if self.session is None:
            self.load()
            
        # Remove background using rembg
        output = remove(image, session=self.session)
        return output

    def is_loaded(self) -> bool:
        return self.session is not None

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        # rembg/u2net uses roughly 150-200MB of VRAM
        return 200

    def unload(self):
        self.session = None
        logger.info("BgRemover: Session cleared")
