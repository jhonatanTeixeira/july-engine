from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.Piper")

VOICES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "voices"))


class PiperModel(BaseModel):
    """
    Piper TTS via subprocess. No GPU, no streaming — returns full WAV bytes.
    Voice files (.onnx + .onnx.json) are downloaded from rhasspy/piper-voices on demand.
    """

    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        os.makedirs(VOICES_DIR, exist_ok=True)

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0

    def load(self, n_ctx=None, num_layers=None):
        pass  # subprocess model — nothing to load into memory

    def is_loaded(self) -> bool:
        return True  # always ready

    def unload(self, model_name=None):
        pass

    def run(self, payload: Dict[str, Any], **kwargs) -> bytes:
        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "default")
        hf_path = payload.get("hf_path")

        if not hf_path:
            from ..persistence import get_backend
            voices = get_backend().get_uploaded_voices()
            info = next((v for v in voices if v.get("id") == voice_id), {})
            hf_path = info.get("piper_path")

        onnx_path = self._ensure_voice(voice_id, hf_path)
        config_path = f"{onnx_path}.json"

        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            import sys
            proc = subprocess.Popen(
                [sys.executable, "-m", "piper", "--model", onnx_path, "--config", config_path, "--output_file", output_path],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate(input=text.encode("utf-8"))

            if proc.returncode != 0:
                raise RuntimeError(f"Piper failed ({proc.returncode}): {stderr.decode()}")

            with open(output_path, "rb") as f:
                return f.read()
        finally:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)

    def _ensure_voice(self, voice_id: str, hf_path: Optional[str]) -> str:
        if hf_path:
            filename = os.path.basename(hf_path)
            local_onnx = os.path.join(VOICES_DIR, filename)
            local_json = f"{local_onnx}.json"

            if not os.path.exists(local_onnx):
                self._download_hf("rhasspy/piper-voices", hf_path, local_onnx)
            if not os.path.exists(local_json):
                self._download_hf("rhasspy/piper-voices", f"{hf_path}.json", local_json)

            return local_onnx

        onnx_file = voice_id if voice_id.endswith(".onnx") else f"{voice_id}.onnx"
        onnx_path = os.path.join(VOICES_DIR, onnx_file)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Piper voice not found: {onnx_path}")
        return onnx_path

    @staticmethod
    def _download_hf(repo_id: str, filename: str, dest: str):
        import shutil
        from huggingface_hub import hf_hub_download

        logger.info(f"Piper: Downloading {filename} from {repo_id}")
        dl = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=VOICES_DIR, local_dir_use_symlinks=False)
        if dl != dest and os.path.exists(dl):
            shutil.move(dl, dest)
