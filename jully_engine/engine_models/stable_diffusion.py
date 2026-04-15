from __future__ import annotations
import logging
import os
from PIL import Image
from typing import Optional, TYPE_CHECKING, List, Dict, Any, Union

if TYPE_CHECKING:
    import torch
    import cv2
    import numpy as np
    from diffusers import StableDiffusionPipeline, LCMScheduler
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    from insightface.app import FaceAnalysis

logger = logging.getLogger("JulyEngine.Models.StableDiffusion")

class StableDiffusion:
    def __init__(self, model_path: str, lora_path: str, backend: str = 'gpu'):
        import torch
        self.backend = backend.lower()
        self.device = "cuda" if self.backend == 'gpu' and torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = None
        self.face_app = None
        self.model_path = model_path
        self.lora_path = lora_path

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o SD 1.5 + IP-Adapter."""
        if self.device == "cpu":
            return 0
        
        # Este modelo usa enable_model_cpu_offload por padrão se estiver na GPU
        # O pico de ativação é o VAE Decode (~200MB além do peso estático)
        # Mas para garantir fluidez, reservamos um bloco maior.
        return 3500 # ~3.5GB para o setup completo

    def load_insightface(self):
        """Inicializa o extrator biométrico 512-d na CPU"""
        from insightface.app import FaceAnalysis
        logger.info("Carregando motor biométrico InsightFace (CPU)...")
        self.face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))


    def load_pipeline(self, model_path: str, lora_path: str):
        from diffusers import StableDiffusionPipeline, LCMScheduler
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        
        logger.info(f"Carregando motor SD 1.5 LCM na {self.device.upper()}...")

        try:
            # 1. Modelo base
            self.pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                local_files_only=True
            )

            # 2. Scheduler LCM
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

            # 3. Patch do feature_extractor
            processor = CLIPImageProcessor()
            self.pipe.feature_extractor = processor
            self.pipe.image_processor = processor

            # 4. LoRAs — SEM fuse_lora() agora, vai fusar depois
            self.pipe.load_lora_weights(lora_path, weight_name="lcm_lora_sd15.safetensors", adapter_name="lcm")
            self.pipe.load_lora_weights(lora_path, weight_name="ip-adapter-faceid_sd15_lora.safetensors", adapter_name="faceid")
            self.pipe.set_adapters(["lcm", "faceid"], adapter_weights=[1.0, 1.0])

            # 5. Image Encoder ANTES do load_ip_adapter
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=self.dtype
            ).to(self.device)
            self.pipe.image_encoder = image_encoder

            # 6. IP-Adapters
            self.pipe.load_ip_adapter(
                ["h94/IP-Adapter", "h94/IP-Adapter-FaceID"],
                subfolder=["models", None],
                weight_name=["ip-adapter-plus_sd15.bin", "ip-adapter-faceid_sd15.bin"]
            )

            # 7. fuse_lora() SOMENTE APÓS load_ip_adapter
            # fuse_lora com lora_scale explícito para não afetar os IP adapters
            self.pipe.fuse_lora(lora_scale=1.0)

            # 8. Mover para device e otimizações de VRAM
            self.pipe.to(self.device)

            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_model_cpu_offload()

            logger.info("Motor multimodal SD/LCM/FaceID pronto!")

        except Exception as e:
            logger.error(f"Falha fatal ao inicializar o motor SD: {e}")
            raise

    def _get_clip_image_embeds(self, pil_image: Image.Image) -> "torch.Tensor":
        """Extrai embedding CLIP para o IP-Adapter Plus."""
        import torch
        # O pipe tem o image_encoder e o feature_extractor carregados
        clip_input = self.pipe.feature_extractor(
            images=pil_image,
            return_tensors="pt"
        ).pixel_values.to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            image_embeds = self.pipe.image_encoder(clip_input).image_embeds  # [1, 768]

        # Plus espera [1, num_tokens, dim] — unsqueeze para adicionar dim de tokens
        return image_embeds.unsqueeze(1)  # [1, 1, 768]


    def _get_faceid_embeds(self, pil_image: Image.Image) -> Optional["torch.Tensor"]:
        """Extrai embedding biométrico 512-d para o IP-Adapter FaceID."""
        import cv2
        import numpy as np
        import torch
        face_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(face_bgr)

        if not faces:
            logger.warning("Nenhum rosto detectado na face_image.")
            return None

        emb = torch.tensor(faces[0].normed_embedding, dtype=self.dtype)
        return emb.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 512]

    def generate(self,
                prompt: str,
                negative_prompt: str = "",
                reference_image: Optional[Image.Image] = None,
                face_image: Optional[Image.Image] = None,
                steps: int = 6,
                cfg_scale: float = 1.5,
                width: int = 512,
                height: int = 512) -> Image.Image:

        if self.pipe is None:
            self.load_insightface()
            self.load_pipeline(self.model_path, self.lora_path)

        import torch

        # --- 1. Pré-computar todos os embeddings ---
        plus_embeds = None
        face_embeds = None

        if reference_image is not None:
            plus_embeds = self._get_clip_image_embeds(reference_image)

        if face_image is not None:
            face_embeds = self._get_faceid_embeds(face_image)

        # --- 2. Definir escalas (ordem = ordem do load_ip_adapter) ---
        # [Plus, FaceID]
        ip_scale = [
            0.6 if plus_embeds is not None else 0.0,
            0.8 if face_embeds is not None else 0.0,
        ]
        self.pipe.set_ip_adapter_scale(ip_scale)

        # --- 3. Montar ip_adapter_image_embeds ---
        # Deve ser lista com um tensor por adapter, na mesma ordem do load_ip_adapter.
        # Se um adapter está desativado (scale=0), passa tensor de zeros com shape correto.
        if plus_embeds is None:
            plus_embeds = torch.zeros(1, 1, 768, dtype=self.dtype, device=self.device)
        if face_embeds is None:
            face_embeds = torch.zeros(1, 1, 512, dtype=self.dtype, device=self.device)

        ip_adapter_image_embeds = [plus_embeds, face_embeds]

        # --- 4. Geração ---
        generator = torch.Generator(device=self.device).manual_seed(42)

        import torch
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                # ip_adapter_image NÃO passa aqui — tudo vai via embeds
            )

        return result.images[0]
    
    def is_loaded(self):
        return self.pipe is not None

    def unload(self):
        """Limpa 100% da memória de vídeo quando a foto acabar."""
        if self.pipe:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            del self.pipe.image_encoder
            del self.pipe
            self.pipe = None
            
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()