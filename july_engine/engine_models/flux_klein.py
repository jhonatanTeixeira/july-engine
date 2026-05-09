import os
import gc
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image

logger = logging.getLogger("JulyEngine.Models.FluxKleinSDNQ")

class FluxKleinNode:
    def __init__(self, backend="gpu", model_meta=None):
        self.backend = backend
        self.meta = model_meta or {}
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.model_t2i = None
        self.model_i2i = None
        self.device = "cuda" if backend == "gpu" else "cpu"
        self.lora_loaded = False
        self.lora_path = os.path.join(os.getcwd(), "models", "ExcellentFullNude_F2K4B_1.safetensors")
        
        self.model_id = self.meta.get("id", "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic")

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o FLUX.2 Klein SDNQ baseada no offload."""
        if self.backend == "cpu":
            return 0
            
        offload = os.environ.get("FLUX_OFFLOAD", None).lower()
        if offload == "sequential":
            return 1000
        elif offload == "cpu":
            return 1500
        else:
            return 3500

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if self.is_loaded():
            return

        logger.info(f"FluxKleinNode: Inicializando carga do FLUX.2 Klein SDNQ (Dinâmico 4/5-bit)...")
        
        import torch
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model
        
        gc.collect()
        torch.cuda.empty_cache()

        # 1. Carrega o Pipeline Base (Os pesos já vêm comprimidos)
        logger.info(f"FluxKleinNode: Montando pipeline com os pesos SDNQ...")
        self.model_t2i = AutoPipelineForText2Image.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir
        )

        # 2. Verifica a disponibilidade do Triton (A trava de segurança para o Windows)
        if torch.cuda.is_available() and triton_is_available:
            logger.info("FluxKleinNode: Triton detectado. Aplicando matmul otimizado...")
            self.model_t2i.transformer = apply_sdnq_options_to_model(self.model_t2i.transformer, use_quantized_matmul=True)
            try:
                self.model_t2i.text_encoder = apply_sdnq_options_to_model(self.model_t2i.text_encoder, use_quantized_matmul=True)
            except Exception:
                pass
        else:
            logger.info("FluxKleinNode: Triton Ausente (Windows). Usando Eager Mode do PyTorch para a matemática. VRAM continua salva!")

        # 3. Clona os ponteiros para a Edição
        logger.info("FluxKleinNode: Clonando pipeline para suporte Img2Img (Zero VRAM overhead)...")
        self.model_i2i = AutoPipelineForImage2Image.from_pipe(self.model_t2i)

        # 4. Kit Sobrevivência 4GB
        logger.info("FluxKleinNode: Ativando travas de segurança de VRAM...")
        offload = os.environ.get("FLUX_OFFLOAD", "sequential").lower()
        
        if offload == "cpu":
            logger.info("FluxKleinNode: Ativando model_cpu_offload...")
            self.model_t2i.enable_model_cpu_offload()
        elif offload == "sequential":
            logger.info("FluxKleinNode: Ativando sequential_cpu_offload...")
            self.model_t2i.enable_sequential_cpu_offload()
        else:
            logger.info("FluxKleinNode: Offload desativado. Usando VRAM completa.")
            self.model_t2i.to(self.device)

        if os.environ.get("ENABLE_VAE", "true").lower() == "true":
            logger.info("FluxKleinNode: Ativando otimizações de VAE (tiling/slicing)...")
            self.model_t2i.vae.enable_tiling()
            self.model_t2i.vae.enable_slicing()
        
        logger.info("FluxKleinNode: Carga finalizada com sucesso!")

    def run(self, payload: Dict[str, Any], **kwargs):
        if not self.is_loaded():
            self.load()

        import torch

        prompt = payload.get("prompt", "")
        width = int(payload.get("width", 512))
        height = int(payload.get("height", 512))
        guidance_scale = float(payload.get("guidance_scale", 1.0)) 
        num_inference_steps = int(payload.get("num_inference_steps", 4))
        seed = int(payload.get("seed", -1)) 
        
        input_image_data = payload.get("image")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # NSFW LoRA Logic - Easter Egg
        headers = payload.get("headers", {})
        nsfw_requested = str(headers.get("x-nsfw", "0")) == "1"

        if nsfw_requested and not self.lora_loaded:
            if os.path.exists(self.lora_path):
                logger.info(f"FluxKleinNode: [NSFW] Ativando LoRA: {self.lora_path}")
                self.model_t2i.load_lora_weights(self.lora_path)
                self.lora_loaded = True
            else:
                logger.warning(f"FluxKleinNode: [NSFW] Arquivo LoRA não encontrado em {self.lora_path}")
        elif not nsfw_requested and self.lora_loaded:
            logger.info("FluxKleinNode: [NSFW] Desativando LoRA (Voltando ao padrão)...")
            self.model_t2i.unload_lora_weights()
            self.lora_loaded = False
        
        # Argumentos básicos aceitos por ambos os pipelines
        kwargs_infer = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator
        }

        try:
            if input_image_data:
                logger.info(f"FluxKleinNode: Modo EDIÇÃO (Img2Img) ativado. Target: {width}x{height}")
                
                if isinstance(input_image_data, str):
                    if "base64," in input_image_data:
                        input_image_data = input_image_data.split("base64,")[1]
                    image_bytes = base64.b64decode(input_image_data)
                    init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                else:
                    init_image = input_image_data
                
                logger.info(f"FluxKleinNode: Original init_image size: {init_image.width}x{init_image.height}")
                    
                # Flux requires multiples of 8 or 16 for stability
                width = (width // 8) * 8
                height = (height // 8) * 8
                
                if init_image.width != width or init_image.height != height:
                    logger.info(f"FluxKleinNode: Resizing init_image to match target {width}x{height}")
                    init_image = init_image.resize((width, height), Image.LANCZOS)
                
                kwargs_infer["image"] = init_image
                kwargs_infer["width"] = width
                kwargs_infer["height"] = height
                
                # Executa o pipeline Img2Img Clonado
                result_image = self.model_i2i(**kwargs_infer).images[0]
                logger.info(f"FluxKleinNode: Result size: {result_image.width}x{result_image.height}")
            else:
                logger.info(f"FluxKleinNode: Modo GERAÇÃO (Text2Img) ativado. Tamanho: {width}x{height}")
                kwargs_infer["width"] = width
                kwargs_infer["height"] = height
                
                result_image = self.model_t2i(**kwargs_infer).images[0]

            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"FluxKleinNode: Erro fatal na inferência: {e}")
            raise e

    def unload(self, model_name: Optional[str] = None):
        if self.is_loaded():
            import torch
            logger.info("FluxKleinNode: Descarregando modelo e limpando VRAM...")
            del self.model_t2i
            del self.model_i2i
            self.model_t2i = None
            self.model_i2i = None
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()

    def is_loaded(self):
        return self.model_t2i is not None

# ==============================================================================
# BLOCO DE TESTE STANDALONE
# ==============================================================================
if __name__ == "__main__":
    logger.info("=== Iniciando Testes do FLUX.2 Klein SDNQ ===")
    from dotenv import load_dotenv
    load_dotenv()
    
    node = FluxKleinNode()
    
    logger.info("\n--- TESTE 1: Text2Image ---")
    payload_t2i = {
        "prompt": "A highly detailed portrait of a cyberpunk knight in neon armor",
        "width": 512,
        "height": 512,
        "num_inference_steps": 4,
        "seed": 1337
    }
    
    b64_out_t2i = None
    try:
        b64_out_t2i = node.run(payload_t2i)
        with open("teste_geracao_sdnq.png", "wb") as fh:
            fh.write(base64.b64decode(b64_out_t2i))
        logger.info("✅ Teste 1 concluído! Imagem salva como 'teste_geracao_sdnq.png'.")
    except Exception as e:
        logger.error(f"❌ Teste 1 Falhou: {e}")

    # Só executa a edição se a geração foi um sucesso
    if b64_out_t2i:
        logger.info("\n--- TESTE 2: Image2Image (Edição Encandeada) ---")
        payload_i2i = {
            "prompt": "The exact same cyberpunk knight, but now standing in a beautiful lush forest with glowing blue mushrooms at night",
            "image": b64_out_t2i, # Passando o Base64 que a IA acabou de cuspir!
            "width": 512,         # Reduzindo para 512x512 para salvar o VAE Encoder
            "height": 512,
            "num_inference_steps": 4,
            "seed": 42
        }
        
        try:
            b64_out_i2i = node.run(payload_i2i)
            with open("teste_edicao_sdnq.png", "wb") as fh:
                fh.write(base64.b64decode(b64_out_i2i))
            logger.info("✅ Teste 2 concluído! A imagem foi editada e salva como 'teste_edicao_sdnq.png'.")
        except Exception as e:
            logger.error(f"❌ Teste 2 Falhou: {e}")
    else:
        logger.warning("⚠️ Teste 2 pulado porque o Teste 1 não gerou a imagem base.")

    node.unload()
    logger.info("=== Testes finalizados com sucesso ===")