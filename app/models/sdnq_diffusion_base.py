import os
import gc
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.SDNQDiffusionBase")


class SDNQDiffusionModel(BaseModel):
    """
    Shared lifecycle for local diffusers pipelines loaded from SDNQ-quantized
    HF repos (Wan2.2 T2V/I2V, FLUX.2 Klein, Qwen-Image-Edit, LTX-2, ...).

    Subclasses set DEFAULT_MODEL_ID / OFFLOAD_ENV_VAR / VRAM_TIERS (and
    PIPELINE_ATTRS if they hold more than one pipeline object, e.g. a
    text2img + img2img pair sharing weights), implement load() (building
    the pipeline(s) and calling self._apply_offload(pipeline_obj)), and
    their own run() with the model-specific payload contract.
    """

    DEFAULT_MODEL_ID: str = ""
    OFFLOAD_ENV_VAR: str = "SDNQ_OFFLOAD"
    VRAM_TIERS: Dict[str, int] = {"sequential": 2000, "cpu": 3000, "none": 6000}
    PIPELINE_ATTRS: Tuple[str, ...] = ("pipeline",)

    def __init__(self, backend="gpu", model_meta=None):
        super().__init__(backend, model_meta)
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.device = "cuda" if backend == "gpu" else "cpu"
        self.model_id = self.meta.get("id", self.DEFAULT_MODEL_ID)
        self._inference_lock = asyncio.Lock()
        for attr in self.PIPELINE_ATTRS:
            setattr(self, attr, None)

    def is_loaded(self) -> bool:
        return getattr(self, self.PIPELINE_ATTRS[0], None) is not None

    def unload(self, model_name: Optional[str] = None):
        if not self.is_loaded():
            return

        import torch

        logger.info(f"{self.__class__.__name__}: Descarregando modelo e limpando VRAM...")
        for attr in self.PIPELINE_ATTRS:
            setattr(self, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    def _vram_table(self) -> Dict[str, int]:
        return self.VRAM_TIERS

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu":
            return 0
        offload = os.environ.get(self.OFFLOAD_ENV_VAR, "sequential").lower()
        table = self._vram_table()
        return table.get(offload, table.get("none", 0))

    def _apply_offload(self, pipeline_obj):
        offload = os.environ.get(self.OFFLOAD_ENV_VAR, "sequential").lower()
        if offload == "cpu":
            logger.info(f"{self.__class__.__name__}: Ativando model_cpu_offload...")
            pipeline_obj.enable_model_cpu_offload()
        elif offload == "sequential":
            logger.info(f"{self.__class__.__name__}: Ativando sequential_cpu_offload...")
            pipeline_obj.enable_sequential_cpu_offload()
        else:
            logger.info(f"{self.__class__.__name__}: Offload desativado. Usando VRAM completa.")
            pipeline_obj.to(self.device)

    async def _stream_file(self, path: str, chunk_size: int = 1 << 20) -> AsyncGenerator[bytes, None]:
        """Streams a file's bytes in fixed-size chunks, deleting it once exhausted.

        Used by video-producing subclasses so a full rendered clip never has to be
        base64-encoded and held in memory as a single blob before being returned.
        """
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
