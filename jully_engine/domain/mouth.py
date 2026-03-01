import logging
import os
import json
from typing import Any, Dict, Optional
from ..engine_models.xtts2 import XTTS2
from ..engine_models.piper import Piper
from ..engine_models.llm_api import LLMApi

logger = logging.getLogger("JulyEngine.Domain.Mouth")

class Mouth:
    """
    Handles Text-to-Speech (TTS) logic and voice resolution.
    Strategies: XTTS2 (cpu, gpu), Piper (cpu, gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()
        self.voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "voices"))

    def _get_strategy(self):
        if self.model_tag == "xtts":
            return XTTS2(backend=self.backend)
        elif self.model_tag == "piper":
            return Piper(backend=self.backend)
        elif self.backend == "api":
            return LLMApi(backend=self.backend)
        else:
            raise ValueError(f"Mouth: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    def _resolve_voice(self, voice_id: str) -> Dict[str, Any]:
        """
        Resolves a voice ID to its configuration by checking voices.json and uploaded_voices.json.
        """
        config_files = ["voices.json", "uploaded_voices.json"]
        for filename in config_files:
            config_path = os.path.join(self.voices_dir, filename)
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        voices = json.load(f)
                        for v in voices:
                            if v.get("id") == voice_id:
                                return v
                except Exception as e:
                    logger.error(f"Mouth: Error reading {filename}: {e}")
        
        # If voice_id looks like a Piper path (e.g. en_US-lessac-medium or contains slashes)
        if "-" in voice_id or "/" in voice_id:
            # Try to construct a piper path if it doesn't have one
            # rhasspy/piper-voices structure: <lang>/<locale>/<name>/<quality>/<file>.onnx
            # This is a bit complex to guess perfectly, so we'll just use it as is if it looks like a full path
            # Or if it's a known short name, we can try to expand it.
            if voice_id == "en_US-lessac-medium":
                return {
                    "id": voice_id,
                    "language": "en",
                    "piper_path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
                }
        
        # Default fallback (yuni or first in voices.json)
        logger.warning(f"Mouth: Voice ID '{voice_id}' not found, using fallback.")
        return {"id": "yuni", "language": "pt", "path": "yuni.wav", "piper_path": "pt/pt_BR/yuni/medium/pt_BR-yuni-medium.onnx"}

    async def speak(self, text: str, voice_id: str, language: Optional[str] = None, output_path: str = "temp.wav"):
        voice_info = self._resolve_voice(voice_id)
        lang = language or voice_info.get("language", "en")
        
        if isinstance(self._strategy, XTTS2):
            # Resolve relative speaker path
            rel_path = voice_info.get("path")
            full_voice_path = os.path.join(self.voices_dir, rel_path)
            return self._strategy.run(text, full_voice_path, lang, output_path)
            
        elif isinstance(self._strategy, Piper):
            # Piper might use the piper_path for downloading from HF or local onnx
            hf_path = voice_info.get("piper_path")
            return self._strategy.run(text, voice_id, output_path, hf_path=hf_path)
            
        elif isinstance(self._strategy, LLMApi):
            # litellm speech implementation
            # litellm uses litellm.speech which is currently OpenAI-compatible
            base_url = None # can be extracted if passed in request
            audio_content = self._strategy.run_tts(self.model_tag, text, voice_id, base_url=base_url)
            if audio_content:
                with open(output_path, "wb") as f:
                    f.write(audio_content)
                return output_path
        
        return None
