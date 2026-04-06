from __future__ import annotations
import gc
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, List, Any

if TYPE_CHECKING:
    import torch
    from diffusers import MotionAdapter, AnimateDiffPipeline, LCMScheduler
    from diffusers.utils import export_to_gif

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
        res = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {label} | Alocado: {used:.2f} GB | Reservado: {res:.2f} GB")

# ---------------------------------------------------------------------------
# Classe principal do Motor de Vídeo (Extremo Offload)
# ---------------------------------------------------------------------------
class LCMVideoPipeline:
    def __init__(
        self,
        base_model_path: str = "models/coldfleshRealisticLCM_v10.safetensors",
        device: str = "cuda",
        dtype: "torch.dtype" = None, # Will handle inside
    ):
        import torch
        if dtype is None:
            dtype = torch.float16
        self.base_model_path = base_model_path
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self._loaded = False

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o motor de vídeo (AnimateDiff + LCM)."""
        if self.device == "cpu":
            return 0
            
        # Com Extreme Offload (CPU Offload) conseguimos rodar em ~4GB.
        # Reservamos 3800MB para segurança na 3050.
        return 3800

    def load(self):
        if self._loaded:
            return

        from diffusers import MotionAdapter, AnimateDiffPipeline, LCMScheduler
        
        print(">> [1/4] Baixando/Carregando Motion Adapter LCM (Aguarde, tem ~1.5GB)...")
        # Usamos o adaptador específico para LCM para manter a geração em apenas 6 passos!
        adapter = MotionAdapter.from_pretrained(
            "wangfuyun/AnimateLCM", 
            torch_dtype=self.dtype
        )

        print(f">> [2/4] Carregando pipeline base: {self.base_model_path}...")
        self.pipe = AnimateDiffPipeline.from_single_file(
            self.base_model_path,
            motion_adapter=adapter,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        print(">> [3/4] Configurando LCMScheduler...")
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config, 
            beta_schedule="linear"
        )

        print(">> [4/4] Ligando aparelhos de suporte à vida (Extremo Offload 4GB)...")
        # 1. Decodifica os frames do vídeo um por um, salvando VRAM massivamente
        self.pipe.enable_vae_slicing()
        
        # 2. Divide as imagens grandes em blocos no VAE
        self.pipe.enable_vae_tiling()
        
        # 3. O SEGREDO DO LABORATÓRIO: Joga os pesos pra RAM e só puxa o que for usar
        self.pipe.enable_model_cpu_offload()

        self._loaded = True
        print_vram("Motor de Vídeo Carregado")
        print(">> Pipeline de Vídeo pronto para o laboratório!")

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,     # 2 segundos de vídeo a 8 FPS
        width: int = 384,         # RESOLUÇÃO ESTRANGULADA PARA NÃO CRASHAR
        height: int = 384,        
        steps: int = 6,           # A mágica do LCM ataca novamente
        guidance_scale: float = 1.5,
        seed: int = 42
    ) -> list:
        
        if not self._loaded:
            self.load()

        import torch
        with torch.inference_mode():
            print_vram("Iniciando inferência 3D")
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # A execução vai ser lenta porque o CPU Offload joga tudo pro barramento da placa mãe
            print(f">> Gerando {num_frames} frames ({width}x{height}) em {steps} passos...")
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            )

            print_vram("Inferência Finalizada")
            return output.frames[0]

    def unload(self):
        self.pipe = None
        self._loaded = False
        free_vram()
        print(">> Motor de vídeo descarregado.")

# ---------------------------------------------------------------------------
# Teste de Laboratório (Execute o arquivo diretamente)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from diffusers.utils import export_to_gif
    import torch
    # Garante que o VRAM esteja limpo antes de começar a loucura
    free_vram()
    
    video_engine = LCMVideoPipeline(
        base_model_path="models/coldfleshRealisticLCM_v10.safetensors"
    )
    
    video_engine.load()

    prompt = "1girl, realistic, highly detailed, cinematic lighting"
    n_prompt = "bad anatomy, deformed, ugly, extra limbs, lowres, blurry"

    frames = video_engine.generate_video(
        prompt=prompt,
        negative_prompt=n_prompt,
        num_frames=16,   # Não aumente isso na 3050!
        width=384,       # Se a 3050 chorar, abaixe para 256
        height=384,
        steps=4
    )

    output_path = "july_lab_video.gif"
    print(f">> Exportando vídeo para {output_path}...")
    export_to_gif(frames, output_path)
    print(">> TESTE FINALIZADO COM SUCESSO!")
    
    video_engine.unload()