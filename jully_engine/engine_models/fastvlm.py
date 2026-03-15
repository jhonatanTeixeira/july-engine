import os
import logging
from typing import Any, Dict
from PIL import Image
import io
import base64

logger = logging.getLogger("JulyEngine.Models.FastVLM")

class FastVLM:
    """
    Model class for FastVLM using Optimum/ONNX.
    """
    def __init__(self, backend="cpu"):
        self.backend = backend
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_id = os.environ.get("VISION_MODEL_ONNX", "onnx-community/FastVLM-0.5B-ONNX")

    def load(self):
        if self.model is None:
            try:
                from optimum.onnxruntime import ORTModelForVision2Seq
                from transformers import AutoProcessor, AutoTokenizer
                
                logger.info(f"FastVLM: Loading model {self.model_id} on {self.backend}")
                self.model = ORTModelForVision2Seq.from_pretrained(self.model_id, provider="CPUExecutionProvider")
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                logger.info("FastVLM loaded successfully.")
            except Exception as e:
                logger.error(f"FastVLM: Failed to load: {e}")
                raise e

    def run(self, payload: Dict[str, Any]) -> str:
        if self.model is None:
            self.load()
            
        image_data = payload.get('image')
        instruction = payload.get('prompt', 'Describe this image briefly:')
        
        if not image_data:
            raise ValueError("No image data provided to FastVLM")

        # Decode Image
        if isinstance(image_data, str) and image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, str):
            img_bytes = base64.b64decode(image_data)
        else:
            img_bytes = image_data
            
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        prompt_text = f"<image>\n{instruction}"
        inputs = self.processor(text=prompt_text, images=img, return_tensors="pt")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        description = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        clean_description = description.replace(instruction, "").strip()
        clean_description = clean_description.replace("<image>", "").strip()
        
        logger.info(f"Engine FastVLM executed successfully on {self.backend} with {self.model_id}")
        return clean_description
