from csv import Error
import os
from typing import Dict, Optional
import logging

import replicate
import replicate.client


logger =  logging.getLogger("JulyEngine.Models.Replicate")

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
            from ..persistence import get_backend
            uploaded_voices = get_backend().get_uploaded_voices()
            voice_info = next((v for v in uploaded_voices if v.get("id") == voice), {})
            
        model = self._extract_model(model)

        language = voice_info.get('language', 'en')

        # The 'voice' argument is the ID string (e.g. "yuni"). We need the actual file path from voice_info (e.g. "yuni.wav")
        rel_path = voice_info.get('path')
        full_path = os.path.join(self.voices_dir, rel_path) if rel_path else None
        
        if not full_path or not os.path.exists(full_path):
            logger.error(f"Replicate: Reference audio not found at {full_path} for voice ID '{voice}'")
            raise Error('Audio path does not exists')

        if model.find('chatterbox') > 0:
            # Replicate input dictionary
            input_data = {
                "text": text,
                "cfg_weight": self._extract_tts_cfg_weight(headers) if language in ['pt', 'en'] else 0.0,
                "language": language,
                "exaggeration": self._extract_tts_exaggeration(headers),
                "temperature": self._extract_tts_temperature(headers),
                "reference_audio": open(full_path, 'rb')
            }
        elif model.find('xtts-v2') > 0:
            input_data = {
                'text': text,
                'speaker': open(full_path, 'rb'),
                'language': language
            }

        # Log the payload being sent (masking the file object)
        log_data = input_data.copy()
        if "reference_audio" in log_data:
            log_data["reference_audio"] = f"<File pointer to {full_path}>"
        
        logger.info(f"Replicate TTS Payload: {log_data}")

        response = replicate.client.Client(api_token=self._extract_api_key(headers)).run(
            model,
            input=input_data
        )

        logger.info(f"Engine Replicate executed successfully on {self.backend} with {model}")
        return response.read()        
        