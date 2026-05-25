import os
import gc
import base64
import logging
import tempfile
from typing import Optional, Dict, Any

try:
    from .base_model import BaseModel
except ImportError:
    from base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.Wan2T2V")


class Wan2T2VPipeline(BaseModel):
    def __init__(self, backend="gpu", model_meta=None):
        super().__init__(backend, model_meta)
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.pipeline = None
        self.device = "cuda" if backend == "gpu" else "cpu"
        self.model_id = self.meta.get("id", "Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32")

    def _is_sdnq_model(self) -> bool:
        variant = self.meta.get("variant", "").lower()
        if variant:
            return variant == "sdnq"
        return "sdnq" in self.model_id.lower()

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu":
            return 0

        offload = os.environ.get("WAN_OFFLOAD", "sequential").lower()
        if self._is_sdnq_model():
            if offload == "sequential":
                return 2000
            elif offload == "cpu":
                return 4000
            else:
                return 10000
        else:
            # Wan2.1-1.3B is much smaller
            if offload == "sequential":
                return 2000
            elif offload == "cpu":
                return 3000
            else:
                return 5000

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if self.is_loaded():
            return

        if self._is_sdnq_model():
            self._load_sdnq()
        else:
            self._load_diffusers_native()

    def _load_sdnq(self):
        logger.info("Wan2T2V: Inicializando carga do Wan2.2 T2V SDNQ uint4...")

        import torch
        from diffusers import WanPipeline, AutoencoderKLWan
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Wan2T2V[SDNQ]: Carregando VAE em bfloat16...")
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )

        logger.info("Wan2T2V[SDNQ]: Montando pipeline principal em bfloat16...")
        self.pipeline = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="balanced",
            cache_dir=self.cache_dir,
        )

        if torch.cuda.is_available() and triton_is_available:
            logger.info("Wan2T2V[SDNQ]: Triton detectado. Aplicando matmul otimizado ao transformer...")
            self.pipeline.transformer = apply_sdnq_options_to_model(
                self.pipeline.transformer, use_quantized_matmul=True
            )
            try:
                self.pipeline.text_encoder = apply_sdnq_options_to_model(
                    self.pipeline.text_encoder, use_quantized_matmul=True
                )
            except Exception:
                pass
        else:
            logger.info("Wan2T2V[SDNQ]: Triton ausente. Usando Eager Mode do PyTorch.")

        self._apply_offload()
        logger.info("Wan2T2V[SDNQ]: Carga finalizada com sucesso!")

    def _load_diffusers_native(self):
        logger.info(f"Wan2T2V: Inicializando carga do {self.model_id} (diffusers native)...")

        import torch
        from diffusers import WanPipeline, AutoencoderKLWan

        gc.collect()
        torch.cuda.empty_cache()

        # TF32 em Ampere+ para throughput melhor em matmuls float32
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        logger.info(f"Wan2T2V[Diffusers]: Carregando VAE em {dtype}...")
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )

        logger.info("Wan2T2V[Diffusers]: Montando pipeline principal...")
        self.pipeline = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )

        # VAE slicing: decodifica frames em fatias para economizar VRAM
        try:
            self.pipeline.enable_vae_slicing()
            logger.info("Wan2T2V[Diffusers]: VAE slicing ativado (pipeline).")
        except AttributeError:
            try:
                self.pipeline.vae.enable_slicing()
                logger.info("Wan2T2V[Diffusers]: VAE slicing ativado (vae direto).")
            except Exception:
                logger.info("Wan2T2V[Diffusers]: VAE slicing indisponível, ignorando.")

        # VAE tiling: divide a imagem em tiles para resoluções altas
        try:
            self.pipeline.enable_vae_tiling()
            logger.info("Wan2T2V[Diffusers]: VAE tiling ativado (pipeline).")
        except AttributeError:
            try:
                self.pipeline.vae.enable_tiling()
                logger.info("Wan2T2V[Diffusers]: VAE tiling ativado (vae direto).")
            except Exception:
                logger.info("Wan2T2V[Diffusers]: VAE tiling indisponível, ignorando.")

        # Attention slicing: processa a atenção em fatias
        try:
            self.pipeline.enable_attention_slicing()
            logger.info("Wan2T2V[Diffusers]: Attention slicing ativado.")
        except AttributeError:
            logger.info("Wan2T2V[Diffusers]: Attention slicing indisponível neste pipeline, ignorando.")

        # xformers memory-efficient attention se disponível
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Wan2T2V[Diffusers]: xformers memory-efficient attention ativado.")
        except Exception:
            logger.info("Wan2T2V[Diffusers]: xformers indisponível, usando SDPA nativo do PyTorch.")

        # SDNQ runtime: aplica matmul quantizado ao transformer carregado em bf16
        # Funciona se os layers responderem ao patch (Triton necessário para o kernel otimizado)
        if os.environ.get("WAN_SDNQ", "1").lower() not in ("0", "false", "no"):
            try:
                import torch as _torch
                from sdnq.common import use_torch_compile as _triton_available
                from sdnq.loader import apply_sdnq_options_to_model
                if _torch.cuda.is_available() and _triton_available:
                    logger.info("Wan2T2V[Diffusers]: Aplicando SDNQ runtime ao transformer...")
                    self.pipeline.transformer = apply_sdnq_options_to_model(
                        self.pipeline.transformer, use_quantized_matmul=True
                    )
                    logger.info("Wan2T2V[Diffusers]: SDNQ runtime aplicado com sucesso.")
                    try:
                        self.pipeline.text_encoder = apply_sdnq_options_to_model(
                            self.pipeline.text_encoder, use_quantized_matmul=True
                        )
                        logger.info("Wan2T2V[Diffusers]: SDNQ runtime aplicado ao text_encoder.")
                    except Exception:
                        pass
                else:
                    logger.info("Wan2T2V[Diffusers]: SDNQ requer Triton — indisponível, ignorando.")
            except ImportError:
                logger.info("Wan2T2V[Diffusers]: sdnq não instalado, ignorando.")
            except Exception as e:
                logger.warning(f"Wan2T2V[Diffusers]: SDNQ runtime falhou ({e}), continuando sem ele.")

        # torchao quantização int8 do transformer para reduzir VRAM e aumentar throughput
        # Só aplica se SDNQ não foi usado (evitar dupla quantização)
        if os.environ.get("WAN_SDNQ", "1").lower() in ("0", "false", "no"):
            try:
                from torchao.quantization import quantize_, Int8WeightOnlyConfig
                logger.info("Wan2T2V[Diffusers]: Aplicando torchao Int8WeightOnlyConfig ao transformer...")
                quantize_(self.pipeline.transformer, Int8WeightOnlyConfig())
                logger.info("Wan2T2V[Diffusers]: torchao int8 aplicado com sucesso.")
            except Exception as e:
                logger.info(f"Wan2T2V[Diffusers]: torchao indisponível ({e}), ignorando quantização.")

        # torch.compile no transformer: reduz overhead por fuse de ops
        # INCOMPATÍVEL com CPU offload: reduce-overhead usa CUDA Graphs (endereços estáticos),
        # mas sequential/model offload move tensors CPU↔GPU a cada forward, mudando endereços.
        # Durante o tracing do Dynamo (1ª passagem), parâmetros viram proxies do grafo FX;
        # o hook do accelerate tenta acessar .__dict__ nesses proxies e quebra com 'node' error.
        offload_mode = os.environ.get("WAN_OFFLOAD", "sequential").lower()
        compile_safe = offload_mode not in ("sequential", "cpu")

        if not compile_safe:
            logger.info(
                f"Wan2T2V[Diffusers]: CPU offload ativo ({offload_mode}) — "
                "torch.compile desativado (CUDA Graphs incompatível com accelerate sequential hooks)."
            )
        elif os.environ.get("WAN_COMPILE", "1").lower() not in ("0", "false", "no"):
            try:
                logger.info("Wan2T2V[Diffusers]: Compilando transformer com torch.compile (reduce-overhead)...")
                self.pipeline.transformer = torch.compile(
                    self.pipeline.transformer,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                logger.info("Wan2T2V[Diffusers]: torch.compile aplicado com sucesso.")
            except Exception as e:
                logger.warning(f"Wan2T2V[Diffusers]: torch.compile falhou ({e}), continuando sem compilação.")

        self._apply_offload()
        logger.info("Wan2T2V[Diffusers]: Carga finalizada com sucesso!")

    def _apply_offload(self):
        offload = os.environ.get("WAN_OFFLOAD", "sequential").lower()
        if offload == "cpu":
            logger.info("Wan2T2V: Ativando model_cpu_offload...")
            self.pipeline.enable_model_cpu_offload()
        elif offload == "sequential":
            logger.info("Wan2T2V: Ativando sequential_cpu_offload...")
            self.pipeline.enable_sequential_cpu_offload()
        else:
            logger.info("Wan2T2V: Offload desativado. Usando VRAM completa.")
            self.pipeline.to(self.device)

    def run(self, payload: Dict[str, Any], **kwargs):
        if not self.is_loaded():
            self.load()

        import torch
        from diffusers.utils import export_to_video

        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "") or None
        height = int(payload.get("height", 480))
        width = int(payload.get("width", 832))
        num_frames = int(payload.get("num_frames", 81))
        num_inference_steps = int(payload.get("num_inference_steps", 40))
        guidance_scale = float(payload.get("guidance_scale", 4.0))
        fps = int(payload.get("fps", 16))
        seed = int(payload.get("seed", -1))

        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"Wan2T2V: Gerando vídeo {width}x{height} com {num_frames} frames, {num_inference_steps} steps...")

        call_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # guidance_scale_2 é específico do modelo SDNQ/CFG distilled — não enviar ao diffusers nativo
        if self._is_sdnq_model():
            call_kwargs["guidance_scale_2"] = float(payload.get("guidance_scale_2", 3.0))

        try:
            output = self.pipeline(**call_kwargs)

            frames = output.frames[0]
            logger.info(f"Wan2T2V: {len(frames)} frames gerados. Exportando para MP4...")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                export_to_video(frames, tmp_path, fps=fps)
                with open(tmp_path, "rb") as f:
                    video_bytes = f.read()
                return base64.b64encode(video_bytes).decode("utf-8")
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Wan2T2V: Erro fatal na inferência: {e}")
            raise e

    def unload(self, model_name: Optional[str] = None):
        if self.is_loaded():
            import torch
            logger.info("Wan2T2V: Descarregando modelo e limpando VRAM...")
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()

    def is_loaded(self):
        return self.pipeline is not None


# ==============================================================================
# BLOCO DE TESTE STANDALONE
# ==============================================================================
if __name__ == "__main__":
    import sys
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Teste standalone do Wan2 T2V")
    parser.add_argument("--prompt", type=str, default="A majestic eagle soaring over snow-capped mountains at golden hour, cinematic, 4K")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, ugly")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="teste_wan2_t2v.mp4")
    parser.add_argument("--offload", type=str, default=None, help="Override WAN_OFFLOAD: sequential | cpu | none")
    parser.add_argument(
        "--model",
        type=str,
        default="Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32",
        help="Model ID: SDNQ (padrão) ou Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument("--no-compile", action="store_true", help="Desativar torch.compile no modo diffusers nativo")
    parser.add_argument("--no-sdnq", action="store_true", help="Desativar SDNQ runtime no modo diffusers nativo")
    args = parser.parse_args()

    if args.offload:
        os.environ["WAN_OFFLOAD"] = args.offload
    if args.no_compile:
        os.environ["WAN_COMPILE"] = "0"
    if args.no_sdnq:
        os.environ["WAN_SDNQ"] = "0"

    logger.info("=== Iniciando Teste Standalone do Wan2 T2V ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"WAN_OFFLOAD: {os.environ.get('WAN_OFFLOAD', 'sequential')}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Resolução: {args.width}x{args.height} | Frames: {args.frames} | Steps: {args.steps} | Seed: {args.seed}")

    node = Wan2T2VPipeline(backend="gpu", model_meta={"id": args.model})

    try:
        node.load()

        payload = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "fps": args.fps,
            "seed": args.seed,
        }

        b64_video = node.run(payload)

        with open(args.output, "wb") as fh:
            fh.write(base64.b64decode(b64_video))

        logger.info(f"Vídeo salvo em: {args.output}")

    except Exception as e:
        logger.error(f"Teste falhou: {e}")
        sys.exit(1)
    finally:
        node.unload()

    logger.info("=== Teste finalizado com sucesso ===")
