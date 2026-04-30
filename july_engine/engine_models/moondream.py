from __future__ import annotations
import logging
from PIL import Image
import gc
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("JulyEngine.Models.MoondreamVLM")

class MoondreamVLM:
    def __init__(self, backend='gpu'):
        self.backend = backend.lower()
        self.vlm = None
        self.tokenizer = None

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o Moondream VLM."""
        if self.backend == "cpu":
            return 0
        return 2048 # ~2GB fixo para este modelo 4-bit lightweight

    def load_transformers(self):
        raise Exception("Moondream2 is not implemented yet on this engine version")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        MID = "vikhyatk/moondream2"
        # O Moondream exige que o código remoto seja confiável para carregar a arquitetura customizada
        self.tokenizer = AutoTokenizer.from_pretrained(MID, revision="2024-08-26") # A revisão crava uma versão estável
        
        try:
            if self.backend == 'gpu' and torch.cuda.is_available():
                logger.info("Engatando Moondream2 na GPU (4-bit + SDPA)...")
                
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, 
                    bnb_4bit_use_double_quant=True
                )

                self.vlm = AutoModelForCausalLM.from_pretrained(
                    MID,
                    trust_remote_code=True,
                    quantization_config=quant_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    attn_implementation="sdpa",
                    revision="2024-08-26"
                )
                logger.info("Sucesso! Motor GPU ativo.")
                
            else:
                logger.info("Engatando Moondream2 na CPU...")
                self.vlm = AutoModelForCausalLM.from_pretrained(
                    MID,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    revision="2024-08-26"
                )
                logger.info("Sucesso! Motor CPU ativo.")
                
        except Exception as e:
            logger.error(f"Falha ao carregar o Moondream2: {e}")
            raise

    def run(self, payload: Dict[str, Any]):
        if self.vlm is None:
            self.load_transformers()
            
        image_data = payload.get("image")
        prompt = payload.get("prompt", "Describe this image.")
        
        if not image_data:
            return ""

        if isinstance(image_data, list):
            return self.run_batch(image_data, prompt)

        img = self.decode_image(image_data)
        results = self.run_batch([img], prompt)
        return results[0] if results else ""

    def run_batch(self, images: List[Any], prompt: str):
        if self.vlm is None:
            self.load_transformers()
            
        if not images: return []

        # 1. Carregamos as imagens do disco
        pil_images = []
        for img_data in images:
            if isinstance(img_data, Image.Image):
                pil_images.append(img_data.convert("RGB"))
            else:
                pil_images.append(self.decode_image(img_data))
        
        # 2. Criamos uma lista com o MESMO prompt repetido para cada imagem
        prompts = [prompt] * len(pil_images)

        import torch
        results = []
        with torch.no_grad():
            # 3. O Moondream2 faz toda a mágica do batching de tensores por debaixo dos panos!
            answers = self.vlm.batch_answer(
                images=pil_images,
                prompts=prompts,
                tokenizer=self.tokenizer
            )
            
            # O retorno já é uma lista de strings limpas!
            results = [ans.strip() for ans in answers]
            
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

    def unload(self):
        if self.vlm:
            del self.vlm
            self.vlm = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()