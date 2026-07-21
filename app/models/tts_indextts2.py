import os
import logging
import tempfile
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.IndexTTS2")

DEFAULT_REPO_ID = "IndexTeam/IndexTTS-2"

# IndexTTS 1.5 already needs ~8GB VRAM and v2 adds an embedded emotion model on
# top — this is NOT expected to run comfortably on 4GB-class GPUs. The
# get_required_vram() value below is intentionally honest about that so the
# orchestrator's VRAM gate rejects it cleanly (MemoryError) on small cards
# instead of letting it OOM mid-load.
REQUIRED_VRAM_MB = 8000


class IndexTTS2Model(BaseModel):
    """
    IndexTTS-2 (IndexTeam) — industrial-grade zero-shot TTS with voice cloning
    plus explicit emotion control (reference audio, emotion vector, or alpha-
    blended emotion reference). Heavier than the engine's other TTS backends;
    intended for GPUs with >=8GB VRAM.
    """

    def __init__(self, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.repo_id = self.meta.get("model") or DEFAULT_REPO_ID
        cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.checkpoint_dir = self.meta.get("checkpoint_dir") or os.path.join(cache_root, "indextts2-checkpoints")
        self._model = None

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else REQUIRED_VRAM_MB

    def _ensure_checkpoint(self):
        config_path = os.path.join(self.checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            return

        logger.info(f"IndexTTS2: Downloading checkpoint {self.repo_id} to {self.checkpoint_dir}...")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.repo_id, local_dir=self.checkpoint_dir)

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return

        self._ensure_checkpoint()

        from indextts.infer_v2 import IndexTTS2

        use_fp16 = self.backend == "gpu"
        logger.info(f"IndexTTS2: Loading from {self.checkpoint_dir} (backend={self.backend}, fp16={use_fp16})")
        self._model = IndexTTS2(
            cfg_path=os.path.join(self.checkpoint_dir, "config.yaml"),
            model_dir=self.checkpoint_dir,
            use_fp16=use_fp16,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        logger.info("IndexTTS2 loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        if self._model is not None:
            del self._model
            self._model = None

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    def run(self, payload: Dict[str, Any], **kwargs) -> bytes:
        from ..services.voice_service import voice_service

        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "default")

        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            raise ValueError(f"IndexTTS2: Voice '{voice_id}' not found")
        voice_path, _ = voice_res

        if self._model is None:
            self.load()

        # Optional emotion control — the engine's own extra over plain cloning.
        emotion_voice_id = payload.get("emotion_voice")
        emo_audio_prompt = None
        if emotion_voice_id:
            emo_res = voice_service.get_voice_path(emotion_voice_id)
            if emo_res:
                emo_audio_prompt = emo_res[0]

        infer_kwargs: Dict[str, Any] = dict(
            spk_audio_prompt=voice_path,
            text=text,
            verbose=False,
        )
        if emo_audio_prompt:
            infer_kwargs["emo_audio_prompt"] = emo_audio_prompt
            infer_kwargs["emo_alpha"] = float(payload.get("emo_alpha", 0.9))
        elif payload.get("emo_vector"):
            infer_kwargs["emo_vector"] = payload["emo_vector"]
            infer_kwargs["use_random"] = bool(payload.get("use_random", False))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._model.infer(output_path=tmp_path, **infer_kwargs)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)
