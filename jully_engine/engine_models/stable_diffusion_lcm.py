"""
Stable Diffusion LCM Pipeline com IP-Adapter FaceID Plus
Equivalente ao pipeline do A1111 com:
  - Sampler      : LCM
  - CFG scale    : 1.5
  - Steps        : 10
  - Size         : 512x512
  - Model        : models/coldfleshRealisticLCM_v10.safetensors  (LCM nativo)
  - ControlNet   : ip-adapter_face_id_plus (weight=1.0, pixel_perfect=True)
  - Otimizações  : medvram + xformers equivalentes

DEPENDÊNCIAS:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install xformers --index-url https://download.pytorch.org/whl/cu121
    pip install diffusers transformers accelerate safetensors pillow numpy
    pip install insightface onnxruntime-gpu
    pip install git+https://github.com/tencent-ailab/IP-Adapter.git

ARQUIVOS em models/:
    models/coldfleshRealisticLCM_v10.safetensors
    models/ip-adapter-faceid-plus_sd15.bin
    models/image_encoder/   <- CLIP ViT-H (baixar do h94/IP-Adapter-FaceID no HF)
        config.json
        model.safetensors
"""
from __future__ import annotations
from diffusers.utils import logging as diffusers_logging
diffusers_logging.disable_progress_bar()

import gc
import logging
import os
import warnings
from pathlib import Path
from PIL import Image
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

warnings.filterwarnings("ignore")
logging.getLogger("xformers").setLevel(logging.ERROR)
os.environ["XFORMERS_MORE_DETAILS"] = "0"


# ---------------------------------------------------------------------------
# Utilitários de memória
# ---------------------------------------------------------------------------

def free_vram():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_vram(label: str = ""):
    import torch
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label} | Alocado: {used:.2f} GB | Reservado: {reserved:.2f} GB")


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class LCMFaceIDPipeline:
    """
    Pipeline SD1.5-LCM + IP-Adapter FaceID Plus via wrapper oficial Tencent.

    Usa IPAdapterFaceIDPlus que gerencia internamente:
      - CLIP image encoder  -> clip_embeds
      - InsightFace buffalo_l -> id_embeds (normed_embedding)
      - Injeção correta dos dois embeddings no UNet

    Para geração SEM face, desconecta temporariamente o encoder_hid_proj
    do UNet e chama o pipe base diretamente.
    """

    DEFAULT_BASE_MODEL    = "models/coldfleshRealisticLCM_v10.safetensors"
    DEFAULT_IP_ADAPTER    = "models/ip-adapter-faceid-plus_sd15.bin"
    DEFAULT_IMAGE_ENCODER = "models/image_encoder"
    HF_IMAGE_ENCODER      = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        ip_adapter_path: Optional[str] = None,
        image_encoder_path: Optional[str] = None,
        device: str = "cuda",
        dtype: "torch.dtype" = None, # Will handle inside
        use_xformers: bool = True,
        use_cpu_offload: bool = False,
        use_sequential_offload: bool = False,
        ip_adapter_scale: float = 1.0,
        use_face_id = True,
        use_vae_slicing = False,
        use_vae_tiling = False,
    ):
        import torch
        if dtype is None:
            dtype = torch.float16
        self.use_face_id = use_face_id
        self.use_vae_slicing = use_vae_slicing
        self.use_vae_tiling = use_vae_tiling

        self.base_model_path    = base_model_path  or self.DEFAULT_BASE_MODEL
        self.ip_adapter_path    = ip_adapter_path  or self.DEFAULT_IP_ADAPTER
        self.image_encoder_path = image_encoder_path or self._resolve_image_encoder()
        self.device             = device
        self.dtype              = dtype
        self.use_xformers       = use_xformers
        self.use_cpu_offload    = use_cpu_offload
        self.use_sequential_offload = use_sequential_offload
        self.ip_adapter_scale   = ip_adapter_scale

        self.pipe     = None
        self.ip_model = None
        self._loaded  = False

    def is_loaded(self) -> bool:
        return self._loaded and self.pipe is not None

    def _resolve_image_encoder(self) -> str:
        local = Path(self.DEFAULT_IMAGE_ENCODER)
        if local.is_dir() and (local / "config.json").exists():
            print(f"  [ok] image_encoder local encontrado: {local}")
            return str(local)
        print(f"  [aviso] models/image_encoder não encontrado, usando HuggingFace: {self.HF_IMAGE_ENCODER}")
        return self.HF_IMAGE_ENCODER

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM necessária baseado na configuração de offload."""
        if self.device == "cpu":
            return 0
        
        # Se for usar offload sequencial ou de modelo, o consumo cai drasticamente
        if self.use_sequential_offload or payload.get("use_sequential_offload"):
            return 2100 # ~2.1GB
        if self.use_cpu_offload or payload.get("use_cpu_offload"):
            return 2800 # ~2.8GB
            
        return 4200 # ~4.2GB para SD1.5 + IP-Adapter sem offload

    def load(self):
        if self._loaded:
            return

        print(">> Carregando pipeline base...")
        self._load_base_pipeline()

        print(">> Configurando LCMScheduler...")
        self._configure_scheduler()

        print(">> Aplicando otimizações de VRAM...")
        self._apply_memory_optimizations()

        if self.use_face_id:
            print(">> Carregando IP-Adapter FaceID Plus...")
            self._load_ip_adapter()

            print(">> Inicializando InsightFace...")
            self._load_face_analysis()

        self._loaded = True
        print_vram("Após carregamento completo")
        print(">> Pipeline pronto!")

    def _load_base_pipeline(self):
        from diffusers import StableDiffusionPipeline
        kwargs = dict(
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        p = Path(self.base_model_path)
        if p.is_file() and p.suffix in (".safetensors", ".ckpt"):
            self.pipe = StableDiffusionPipeline.from_single_file(self.base_model_path, **kwargs)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(self.base_model_path, **kwargs)

    def _configure_scheduler(self):
        from diffusers import LCMScheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        print("  [ok] LCMScheduler configurado")

    def _apply_memory_optimizations(self):
        if self.use_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("  [ok] xformers ativado")
            except Exception as e:
                print(f"  [aviso] xformers indisponível: {e}")

        if self.use_vae_slicing:
            self.pipe.enable_vae_slicing()
            print("  [ok] VAE slicing ativado")

        if self.use_vae_tiling:
            self.pipe.enable_vae_tiling()
            print("  [ok] VAE tiling ativado")

        self.pipe.enable_attention_slicing(slice_size="auto")
        print("  [ok] Attention slicing ativado")

        if self.use_sequential_offload:
            self.pipe.enable_sequential_cpu_offload()
            print("  [ok] Sequential CPU offload (~4GB VRAM)")
        elif self.use_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("  [ok] Model CPU offload (~6GB VRAM)")
        else:
            self.pipe = self.pipe.to(self.device)
            print(f"  [ok] Pipeline em {self.device}")

    def _load_ip_adapter(self):
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

        if not Path(self.ip_adapter_path).is_file():
            raise FileNotFoundError(
                f"IP-Adapter não encontrado: {self.ip_adapter_path}\n"
                "Baixe em: https://huggingface.co/h94/IP-Adapter-FaceID"
            )

        self.ip_model = IPAdapterFaceIDPlus(
            sd_pipe=self.pipe,
            image_encoder_path=self.image_encoder_path,
            ip_ckpt=self.ip_adapter_path,
            device=self.device,
            torch_dtype=self.dtype,
        )
        print(f"  [ok] IP-Adapter carregado: {self.ip_adapter_path}")
        print(f"  [ok] Image encoder: {self.image_encoder_path}")

    def _load_face_analysis(self):
        from insightface.app import FaceAnalysis
        self._face_analysis = FaceAnalysis(
            name="buffalo_l",
            providers=[
                # "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ],
        )
        self._face_analysis.prepare(ctx_id=0, det_size=(512, 512))
        print("  [ok] InsightFace carregado")

    # ------------------------------------------------------------------
    # Extração de face
    # ------------------------------------------------------------------

    def extract_face_embeds(self, face_image: Union[Image.Image, "np.ndarray"]) -> "torch.Tensor":
        """Extrai normed_embedding via InsightFace. Retorna Tensor [1, 512]."""
        import numpy as np
        import torch
        if isinstance(face_image, Image.Image):
            img_bgr = np.array(face_image.convert("RGB"))[:, :, ::-1]
        else:
            img_bgr = face_image

        faces = self._face_analysis.get(img_bgr)
        if not faces:
            raise ValueError("Nenhuma face detectada na imagem de referência!")

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return torch.from_numpy(face.normed_embedding).unsqueeze(0)

    def _preprocess_face_image(
        self, image: Union[Image.Image, "np.ndarray"], crop: bool = True
    ) -> Image.Image:
        """Crop na face + resize 512x512 (equivale ao Crop and Resize do ControlNet)."""
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])
        image = image.convert("RGB")

        if crop and hasattr(self, "_face_analysis"):
            img_bgr = np.array(image)[:, :, ::-1]
            faces = self._face_analysis.get(img_bgr)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                px, py = int((x2-x1)*0.3), int((y2-y1)*0.3)
                x1, y1 = max(0, x1-px), max(0, y1-py)
                x2, y2 = min(image.width, x2+px), min(image.height, y2+py)
                image = image.crop((x1, y1, x2, y2))

        return image.resize((512, 512), Image.LANCZOS)

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompt: str,
        face_image: Optional[Union[Image.Image, "np.ndarray"]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 10,
        guidance_scale: float = 1.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        num_images_per_prompt: int = 1,
        ip_adapter_scale: Optional[float] = None,
        shortcut: bool = False,
        crop_face: bool = True,
    ) -> list[Image.Image]:
        """
        Gera imagens com LCM + IP-Adapter FaceID Plus.

        face_image=None -> gera só com prompt (desconecta IP-Adapter temporariamente)
        face_image=PIL  -> usa FaceID Plus normalmente
        """
        if not self._loaded:
            self.load()

        import torch
        with torch.inference_mode():
            scale = ip_adapter_scale if ip_adapter_scale is not None else self.ip_adapter_scale
            print_vram("Antes da inferência")
            
            if not self.use_face_id:
                generator = None

                if seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(seed)

                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    num_images_per_prompt=num_images_per_prompt,
                )
                images = result.images

            elif face_image is not None:
                # ---- COM face ----
                face_pil         = self._preprocess_face_image(face_image, crop=crop_face)
                faceid_embeds    = self.extract_face_embeds(face_pil)

                images = self.ip_model.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    face_image=face_pil,         # usado pelo CLIP encoder interno
                    faceid_embeds=faceid_embeds, # usado pelo InsightFace proj
                    shortcut=shortcut,
                    s_scale=scale,
                    num_samples=num_images_per_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed if seed is not None else -1,
                )

            else:

                dummy_pil    = Image.new('RGB', (512, 512))
                dummy_embeds = torch.zeros((1, 512), dtype=self.dtype, device=self.device)

                images = self.ip_model.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    face_image=dummy_pil,        # CLIP processa o nada
                    faceid_embeds=dummy_embeds,  # Tensor zerado (sem rosto)
                    shortcut=shortcut,
                    s_scale=0.0,                 # A MAGIA: Multiplica o IP-Adapter por ZERO
                    num_samples=num_images_per_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed if seed is not None else -1,
                )
                                
            print_vram("Após inferência")
            free_vram()
            return images

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def unload(self):
        self.pipe = None
        self.ip_model = None
        if hasattr(self, "_face_analysis"):
            del self._face_analysis
        self._loaded = False
        free_vram()
        print(">> Modelos descarregados.")

    def __repr__(self):
        status = "carregado" if self._loaded else "não carregado"
        return f"LCMFaceIDPipeline(model='{Path(self.base_model_path).name}', status={status})"


# ---------------------------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    pipeline = LCMFaceIDPipeline(
        base_model_path="models/coldfleshRealisticLCM_v10.safetensors",
        ip_adapter_path="models/ip-adapter-faceid-plus_sd15.bin",
        # image_encoder_path="models/image_encoder",  # descomente se tiver local
        device="cuda",
        dtype=torch.float16,
        use_xformers=True,
        use_cpu_offload=False,
        ip_adapter_scale=1.0,
    )

    pipeline.load()

    # -----------------------------------------------------------------------
    # Teste COM face
    # -----------------------------------------------------------------------
    face_ref = Image.open(
        Path("E:/projects/jhon/ai/jully/july_dating/resources/pictures/anna_profile.png")
    )

    images = pipeline(
        prompt="1girl, portrait, beautiful, cinematic lighting, detailed skin",
        face_image=face_ref,
        negative_prompt="blurry, deformed, ugly, bad anatomy, lowres",
        num_inference_steps=6,
        guidance_scale=1.5,
        width=512,
        height=512,
        # seed=42,
    )
    images[0].save("output_with_face.png")
    print("Salvo: output_with_face.png")

    # -----------------------------------------------------------------------
    # Teste SEM face
    # -----------------------------------------------------------------------
    images = pipeline(
        prompt="1girl, portrait, beautiful, cinematic lighting, detailed skin",
        negative_prompt="blurry, deformed, ugly, bad anatomy, lowres",
        num_inference_steps=6,
        guidance_scale=1.5,
        width=512,
        height=512,
        # seed=42,
    )
    images[0].save("output_no_face.png")
    print("Salvo: output_no_face.png")

    pipeline.unload()