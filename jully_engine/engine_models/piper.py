import os
import logging
import subprocess
from typing import Any, Dict, Optional
from huggingface_hub import hf_hub_download

logger = logging.getLogger("JulyEngine.Models.Piper")

class Piper:
    """
    Piper TTS model using onnxruntime.
    Backends: cpu, gpu.
    """
    def __init__(self, backend="cpu"):
        self.backend = backend
        self.voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "voices"))
        os.makedirs(self.voices_dir, exist_ok=True)

    def _ensure_voice_files(self, voice_id: str, hf_path: Optional[str] = None) -> str:
        """
        Ensures .onnx and .onnx.json exist. Downloads from HF if missing.
        Returns the local path to the .onnx file.
        """
        if hf_path:
            # e.g. en/en_US/lessac/medium/en_US-lessac-medium.onnx
            filename = os.path.basename(hf_path)
            local_onnx_path = os.path.join(self.voices_dir, filename)
            local_config_path = f"{local_onnx_path}.json"
            
            repo_id = "rhasspy/piper-voices"
            
            if not os.path.exists(local_onnx_path):
                logger.info(f"Piper: Downloading {hf_path} from {repo_id}...")
                try:
                    # Use absolute path for download
                    dl_path = hf_hub_download(repo_id=repo_id, filename=hf_path, local_dir=self.voices_dir, local_dir_use_symlinks=False)
                    # hf_hub_download might return a different path if local_dir structure is used
                    if dl_path != local_onnx_path and os.path.exists(dl_path):
                        import shutil
                        shutil.move(dl_path, local_onnx_path)
                except Exception as e:
                    logger.error(f"Piper: Download failed for {hf_path}: {e}")
                    raise e
            
            if not os.path.exists(local_config_path):
                config_hf_path = f"{hf_path}.json"
                logger.info(f"Piper: Downloading {config_hf_path} from {repo_id}...")
                try:
                    dl_path = hf_hub_download(repo_id=repo_id, filename=config_hf_path, local_dir=self.voices_dir, local_dir_use_symlinks=False)
                    if dl_path != local_config_path and os.path.exists(dl_path):
                        import shutil
                        shutil.move(dl_path, local_config_path)
                except Exception as e:
                    logger.error(f"Piper: Download failed for {config_hf_path}: {e}")
            
            return local_onnx_path
        else:
            # Fallback to simple voice_id check in the root of voices_dir
            if not voice_id.endswith(".onnx"):
                onnx_file = f"{voice_id}.onnx"
            else:
                onnx_file = voice_id
            
            onnx_path = os.path.join(self.voices_dir, onnx_file)
            if not os.path.exists(onnx_path):
                logger.error(f"Piper: Voice file {onnx_path} not found and no hf_path provided.")
                raise FileNotFoundError(f"Piper voice not found: {voice_id}")
            return onnx_path

    def run(self, text: str, voice_id: str, hf_path: Optional[str] = None) -> bytes:
        if not hf_path:
            from ..persistence import get_backend
            uploaded_voices = get_backend().get_uploaded_voices()
            voice_info = next((v for v in uploaded_voices if v.get("id") == voice_id), {})
            hf_path = voice_info.get("piper_path")

        onnx_path = self._ensure_voice_files(voice_id, hf_path)
        # Piper expects the config file in the same dir as onnx, or passed via --config
        config_path = f"{onnx_path}.json"
        
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
                
            logger.info(f"Piper: Synthesizing to {output_path} using model {onnx_path}")
            
            import sys
            command = [
                sys.executable, "-m", "piper",
                "--model", onnx_path,
                "--config", config_path,
                "--output_file", output_path
            ]
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "No stderr"
                raise RuntimeError(f"Piper process failed with code {process.returncode}: {error_msg}")
                
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
                
            os.remove(output_path)
            logger.info(f"Engine Piper executed successfully on {self.backend} with Piper")
            return audio_bytes
        except Exception as e:
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"Piper: execution failed: {e}")
            raise e
