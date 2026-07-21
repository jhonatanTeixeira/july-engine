import os
import gc
import io
import base64
import asyncio
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any
from PIL import Image

try:
    from .sdnq_diffusion_base import SDNQDiffusionModel
except ImportError:
    from sdnq_diffusion_base import SDNQDiffusionModel

logger = logging.getLogger("JulyEngine.Models.LTX2")

_VRAM_TIERS = {"sequential": 1500, "cpu": 3000, "none": 7000}

# LTX-2 output audio has no documented sample rate in the model card at the time
# this was written — 24kHz is a common default for diffusers audio pipelines and
# is exposed as an override via payload["audio_sample_rate"]; verify against the
# real pipeline output before relying on this in production.
_DEFAULT_AUDIO_SAMPLE_RATE = 24000


class LTX2Model(SDNQDiffusionModel):
    DEFAULT_MODEL_ID = "Disty0/LTX-2-SDNQ-4bit-dynamic"
    OFFLOAD_ENV_VAR = "LTX2_OFFLOAD"
    VRAM_TIERS = _VRAM_TIERS

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        if self.is_loaded():
            return

        logger.info(f"LTX2: Inicializando carga de {self.model_id}...")

        import torch
        from diffusers import DiffusionPipeline
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        gc.collect()
        torch.cuda.empty_cache()

        # Generic DiffusionPipeline.from_pretrained auto-resolves the concrete
        # LTX-2 pipeline class via the repo's model_index.json.
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )

        if torch.cuda.is_available() and triton_is_available:
            logger.info("LTX2: Triton detectado. Aplicando matmul otimizado ao transformer...")
            self.pipeline.transformer = apply_sdnq_options_to_model(
                self.pipeline.transformer, use_quantized_matmul=True
            )
        else:
            logger.info("LTX2: Triton ausente. Usando Eager Mode do PyTorch.")

        try:
            self.pipeline.vae.enable_tiling()
        except Exception:
            logger.info("LTX2: VAE tiling indisponível, ignorando.")

        self._apply_offload(self.pipeline)
        logger.info("LTX2: Carga finalizada com sucesso!")

    @staticmethod
    def _decode_image(image_data) -> Optional["Image.Image"]:
        if not image_data:
            return None
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    @staticmethod
    def _write_wav(path: str, audio, sample_rate: int) -> None:
        import wave
        import numpy as np

        audio = np.asarray(audio)
        if audio.dtype.kind == "f":
            audio = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        channels = 1 if audio.ndim == 1 else audio.shape[-1]

        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

    def _mux(self, video_path: str, audio_path: str) -> str:
        import imageio_ffmpeg

        muxed_fd, muxed_path = tempfile.mkstemp(suffix=".mp4")
        os.close(muxed_fd)

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            muxed_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return muxed_path

    def _render(self, payload: Dict[str, Any]) -> str:
        """Blocking render (+ optional audio mux) — runs inside asyncio.to_thread."""
        from diffusers.utils import export_to_video

        image = self._decode_image(payload.get("image"))

        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "") or None
        width = int(payload.get("width", 768))
        height = int(payload.get("height", 512))
        num_frames = int(payload.get("num_frames", 121))
        frame_rate = int(payload.get("frame_rate", 25))
        num_inference_steps = int(payload.get("num_inference_steps", 40))
        guidance_scale = float(payload.get("guidance_scale", 4.0))
        audio_sample_rate = int(payload.get("audio_sample_rate", _DEFAULT_AUDIO_SAMPLE_RATE))

        call_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="np",
        )
        if image is not None:
            call_kwargs["image"] = image

        logger.info(f"LTX2: Gerando vídeo {width}x{height} com {num_frames} frames, {num_inference_steps} steps...")
        video, audio = self.pipeline(**call_kwargs)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        export_to_video(video, video_path, fps=frame_rate)

        if audio is None:
            return video_path

        audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(audio_fd)
        try:
            self._write_wav(audio_path, audio, audio_sample_rate)
            muxed_path = self._mux(video_path, audio_path)
        finally:
            os.unlink(video_path)
            os.unlink(audio_path)

        return muxed_path

    async def run(self, payload: Dict[str, Any], **kwargs):
        if not self.is_loaded():
            await asyncio.to_thread(self.load)

        async with self._inference_lock:
            try:
                tmp_path = await asyncio.to_thread(self._render, payload)
            except Exception as e:
                logger.error(f"LTX2: Erro fatal na inferência: {e}")
                raise

            async for chunk in self._stream_file(tmp_path):
                yield chunk
