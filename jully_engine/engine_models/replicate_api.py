import os
from typing import Dict, Optional

import replicate
import replicate.client

class Replicate:
    def __init__(self):
        self.voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "voices"))

    def _extract_api_key(self, headers):
        return headers.get('x-api-key')
    
    def _extract_tts_exaggeration(self, headers):
        return float(headers.get('x-tts-exageration', '0.5'))
    
    def _extract_tts_temperature(self, headers):
        return float(headers.get('x-tts-temperature', '0.8'))

    def _extract_tts_cfg_weight(self, headers):
        return float(headers.get('x-tts-cfg-weight', '0.5'))
    
    def _extract_model(self, model: str):
        return model.removeprefix('replicate/')

    def run_tts(self, model: str, text: str, voice: str = None, headers: Optional[Dict[str, str]] = None, voice_info=None, **kwargs):
        if not voice_info:
            voice_info = {}

        language = voice_info.get('language', 'en')

        # The 'voice' argument is the ID string (e.g. "yuni"). We need the actual file path from voice_info (e.g. "yuni.wav")
        rel_path = voice_info.get('path')
        full_path = os.path.join(self.voices_dir, rel_path) if rel_path else None

        # Replicate input dictionary
        input_data = {
            "text": text,
            "cfg_weight": self._extract_tts_cfg_weight(headers) if language in ['pt', 'en'] else 0.0,
            "language": language,
            "exaggeration": self._extract_tts_exaggeration(headers),
            "temperature": self._extract_tts_temperature(headers),
        }

        # Only add reference_audio if the file exists, otherwise replicate throws validation error for passing null
        if full_path and os.path.exists(full_path):
            input_data["reference_audio"] = open(full_path, 'rb')
        else:
            import logging
            logging.getLogger("JulyEngine.Models.Replicate").warning(f"Replicate: Reference audio not found at {full_path} for voice ID '{voice}'")

        # Log the payload being sent (masking the file object)
        log_data = input_data.copy()
        if "reference_audio" in log_data:
            log_data["reference_audio"] = f"<File pointer to {full_path}>"
        
        import logging
        logging.getLogger("JulyEngine.Models.Replicate").info(f"Replicate TTS Payload: {log_data}")

        response = replicate.client.Client(api_token=self._extract_api_key(headers)).run(
            self._extract_model(model),
            input=input_data
        )

        return response.read()        
        