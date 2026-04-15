from __future__ import annotations
import logging
from PIL import Image
import gc
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("JulyEngine.Models.FastVLM")

class FastVLM:
    def __init__(self, backend='gpu'):
        self.backend = backend.lower()
        self.vlm = None
        self.tokenizer = None
        self.IMAGE_TOKEN_INDEX = -200

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o FastVLM (0.5B 4-bit)."""
        if self.backend == "cpu":
            return 0
        return 1200 # ~1.2GB para este modelo minúsculo

    def load_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        MID = "apple/FastVLM-0.5B"
        
        # O Tokenizer é leve e agnóstico, carrega igual para ambos
        self.tokenizer = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        
        try:
            if self.backend == 'gpu' and torch.cuda.is_available():
                logger.info("Engatando FastVLM na GPU (4-bit + SDPA)...")
                
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, 
                    bnb_4bit_use_double_quant=True
                )

                self.vlm = AutoModelForCausalLM.from_pretrained(
                    MID,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,     # GPU adora FP16
                    attn_implementation="sdpa",    # Otimização extrema para Ampere
                    low_cpu_mem_usage=True
                )
                
                active_attn = getattr(self.vlm.config, "_attn_implementation", "unknown")
                logger.info(f"Sucesso! Motor GPU ativo. Atenção: {active_attn.upper()}")
                
            else:
                logger.info("Engatando FastVLM na CPU (Precisão Padrão/Float32)...")
                if self.backend == 'gpu':
                    logger.warning("Backend definido como 'gpu', mas CUDA indisponível. Fazendo fallback para CPU.")
                
                self.vlm = AutoModelForCausalLM.from_pretrained(
                    MID,
                    device_map="cpu",              # Força o carregamento na memória RAM
                    trust_remote_code=True,
                    torch_dtype=torch.float32,     # CPU processa Float32 nativamente
                    # Removemos o quant_config (BnB) e o SDPA forçado
                    low_cpu_mem_usage=True
                )
                logger.info("Sucesso! Motor CPU ativo.")
                
        except Exception as e:
            logger.error(f"Falha ao carregar o VLM no backend {self.backend}: {e}")
            raise

    def run(self, payload: Dict[str, Any]):
        if self.vlm is None:
            self.load_transformers()
            
        image_data = payload.get("image")
        prompt = payload.get("prompt", "Describe this image.")
        
        if not image_data:
            return ""

        # Se for uma lista, redireciona para batch
        if isinstance(image_data, list):
            return self.run_batch(image_data, prompt)

        img = self.decode_image(image_data)
        results = self.run_batch([img], prompt)
        return results[0] if results else ""

    def run_batch(self, images: List[Any], prompt: str):
        if self.vlm is None:
            self.load_transformers()
            
        import torch
        batch_len = len(images)
        if batch_len == 0: return []

        # Preparar imagens (converter base64 para PIL se necessário)
        pil_images = []
        for img_data in images:
            if isinstance(img_data, Image.Image):
                pil_images.append(img_data.convert("RGB"))
            else:
                pil_images.append(self.decode_image(img_data))

        chat = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)

        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        b_pre = pre_ids.repeat(batch_len, 1)
        b_img = torch.tensor([[self.IMAGE_TOKEN_INDEX]] * batch_len, dtype=pre_ids.dtype)
        b_post = post_ids.repeat(batch_len, 1)
        
        input_ids = torch.cat([b_pre, b_img, b_post], dim=1).to(self.vlm.device)
        attention_mask = torch.ones_like(input_ids, device=self.vlm.device)

        px_list = []
        for img in pil_images:
            px = self.vlm.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            px_list.append(px)
        
        batched_px = torch.cat(px_list, dim=0).to(self.vlm.device, dtype=self.vlm.dtype)

        with torch.no_grad():
            out = self.vlm.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=batched_px,
                max_new_tokens=128,
                do_sample=False
            )

        results = []
        for i in range(batch_len):
            full_text = self.tokenizer.decode(out[i], skip_special_tokens=True)
            clean_content = full_text.split("ASSISTANT:")[-1] if "ASSISTANT:" in full_text else full_text
            results.append(clean_content.strip())
            
        return results

    def decode_image(self, image_data):
        import base64
        import io
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def is_loaded(self) -> bool:
        return self.vlm is not None

    def unload(self):
        """Limpa a memória RAM e a VRAM ativamente."""
        if self.vlm:
            del self.vlm
            self.vlm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        # Força o coletor de lixo do Python
        gc.collect()
        
        # Se estávamos usando a GPU, esvazia o cache do CUDA para não dar OOM na próxima tarefa
        if torch.cuda.is_available():
            torch.cuda.empty_cache()