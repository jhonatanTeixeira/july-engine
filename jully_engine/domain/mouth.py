import logging
import os
import json
from typing import Any, Dict, Optional

from july_engine.jully_engine.engine_models.replicate import Replicate
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
        if self.backend == "api":
            if self.model_tag.startswith('replicate/'):
                return Replicate()
            
            return LLMApi(backend=self.backend)
        elif self.model_tag == "xtts":
            return XTTS2(backend=self.backend)
        elif self.model_tag == "piper":
            return Piper(backend=self.backend)
        
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
        
        if "-" in voice_id or "/" in voice_id:
            if voice_id == "en_US-lessac-medium":
                return {
                    "id": voice_id,
                    "language": "en",
                    "piper_path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
                }
        
        logger.warning(f"Mouth: Voice ID '{voice_id}' not found, using fallback.")
        return {"id": "yuni", "language": "pt", "path": "yuni.wav", "piper_path": "pt/pt_BR/yuni/medium/pt_BR-yuni-medium.onnx"}

    async def speak(self, payload: Dict[str, Any], output_path: str = "temp.wav"):
        # For local strategies, unpack payload
        text = payload.get("input", payload.get("text", ""))
        voice_id = payload.get("voice", "")
        language = payload.get("language")

        voice_info = self._resolve_voice(voice_id)
        lang = language or voice_info.get("language", "en")

        if isinstance(self._strategy, (LLMApi, Replicate)):
            model = payload.pop("model", self.model_tag)
            text = payload.pop("input", payload.pop("text", ""))
            voice_id = payload.pop("voice", "")
            headers = payload.pop("headers", {})
            payload.setdefault('voice_info', voice_info)
            
            audio_content = self._strategy.run_tts(model, text, voice_id, headers=headers, **payload)
            
            if audio_content:
                with open(output_path, "wb") as f:
                    f.write(audio_content)
                return output_path
            return None
        
        if isinstance(self._strategy, XTTS2):
            rel_path = voice_info.get("path")
            full_voice_path = os.path.join(self.voices_dir, rel_path)
            return self._strategy.run(text, full_voice_path, lang, output_path)
            
        elif isinstance(self._strategy, Piper):
            hf_path = voice_info.get("piper_path")
            return self._strategy.run(text, voice_id, output_path, hf_path=hf_path)
            
        return None
