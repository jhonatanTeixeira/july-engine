import gc
import logging
import asyncio
import uuid
from typing import Any, Dict, List, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.FastVLM")

MODEL_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200


class FastVLMModel(BaseModel):
    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._vlm = None
        self._tokenizer = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 1200

    def load(self, n_ctx=None, num_layers=None):
        if self._vlm is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        if self.backend == "gpu" and torch.cuda.is_available():
            logger.info("FastVLM: Loading on GPU (4-bit NF4 + SDPA)")
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._vlm = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quant,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        else:
            logger.info("FastVLM: Loading on CPU (float32)")
            self._vlm = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        logger.info("FastVLM loaded.")

    def is_loaded(self) -> bool:
        return self._vlm is not None

    def unload(self, model_name=None):
        import torch

        if self._vlm:
            del self._vlm
            self._vlm = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def chat(self, payload: dict):
        """Alias para run() com suporte a payload OpenAI."""
        return await asyncio.to_thread(self.run, payload)

    def run(self, payload: Dict[str, Any], **kwargs):
        if self._vlm is None:
            self.load()

        # Extração de prompt e imagens do payload OpenAI se presente
        messages = payload.get("messages", [])
        images = payload.get("images") or payload.get("image")
        prompt = payload.get("prompt", "")

        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, list):
                extracted_images = []
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            extracted_images.append(url)
                prompt = " ".join(text_parts)
                if not images:
                    images = extracted_images
            else:
                prompt = content
        
        if not prompt:
            prompt = "Describe this image."

        if not images:
            return []

        if isinstance(images, list):
            pil_images = [self._decode(img) for img in images]
        else:
            pil_images = [self._decode(images)]

        results = self._run_batch(pil_images, prompt)
        
        # Formata resposta OpenAI se mensagens estavam presentes
        if messages:
             content = results if isinstance(payload.get("images"), list) else results[0]
             return {
                 "id": f"vlm-{uuid.uuid4().hex[:8]}",
                 "choices": [{
                     "index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"
                 }]
             }

        return results if isinstance(payload.get("images"), list) else results[0]

    def _run_batch(self, pil_images: list, prompt: str) -> List[str]:
        import torch

        batch_len = len(pil_images)
        chat = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = self._tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)

        pre_ids = self._tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self._tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

        b_pre = pre_ids.repeat(batch_len, 1)
        b_img = torch.tensor([[IMAGE_TOKEN_INDEX]] * batch_len, dtype=pre_ids.dtype)
        b_post = post_ids.repeat(batch_len, 1)

        input_ids = torch.cat([b_pre, b_img, b_post], dim=1).to(self._vlm.device)
        attention_mask = torch.ones_like(input_ids, device=self._vlm.device)

        px_list = [
            self._vlm.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            for img in pil_images
        ]
        batched_px = torch.cat(px_list, dim=0).to(self._vlm.device, dtype=self._vlm.dtype)

        with torch.no_grad():
            out = self._vlm.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=batched_px,
                max_new_tokens=128,
                do_sample=False,
            )

        results = []
        logger.info(f"FastVLM: Processing {batch_len} outputs...")

        for i in range(batch_len):
            text = self._tokenizer.decode(out[i], skip_special_tokens=True)
            logger.info(f"FastVLM: Raw output[{i}]: {text[:100]}...")
            clean = text.split("ASSISTANT:")[-1] if "ASSISTANT:" in text else text
            results.append(clean.strip())
        
        return results

    def _decode(self, image_data):
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
