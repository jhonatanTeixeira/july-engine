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
        
        # Get the relative path and build the full path
        rel_path = voice or voice_info.get('path')
        full_path = os.path.join(self.voices_dir, rel_path) if rel_path else None
            
        response = replicate.client.Client(api_token=self._extract_api_key(headers)).run(
            self._extract_model(model),
            input={
                "text": text,
                "reference_audio": open(full_path, 'rb') if full_path and os.path.exists(full_path) else None,
                "cfg_weight": self._extract_tts_cfg_weight(headers) if language in ['pt', 'en'] else 0.0,
                "language": language,
                "exaggeration": self._extract_tts_exaggeration(headers),
                "temperature": self._extract_tts_temperature(headers),
            }
        )
        
        return response.read()
        
        