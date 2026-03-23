import logging
from PIL import Image
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("JulyEngine.Models.MoondreamVLM")

class MoondreamVLM:
    def __init__(self, backend='gpu'):
        self.backend = backend.lower()
        self.vlm = None
        self.tokenizer = None
        self.load_transformers()

    def load_transformers(self):
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

    def create_batch_completion(self, valid_image_paths: list, prompt_text: str):
        if not valid_image_paths: return []

        # 1. Carregamos as imagens do disco
        images = [Image.open(img_path).convert("RGB") for img_path in valid_image_paths]
        
        # 2. Criamos uma lista com o MESMO prompt repetido para cada imagem
        prompts = [prompt_text] * len(images)

        results = []
        with torch.no_grad():
            # 3. O Moondream2 faz toda a mágica do batching de tensores por debaixo dos panos!
            # Muito mais seguro que concatenar matrizes na mão.
            answers = self.vlm.batch_answer(
                images=images,
                prompts=prompts,
                tokenizer=self.tokenizer
            )
            
            # O retorno já é uma lista de strings limpas!
            results = [ans.strip() for ans in answers]
            
        return results

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