from typing import Dict, Optional

import replicate
import replicate.client


class Replicate:
    def _extract_api_key(headers):
        return headers.get('x-api-key')
    
    def _extract_tts_exaggeration(headers):
        return float(headers.get('x-tts-exageration', '0.5'))
    
    def _extract_tts_temperature(headers):
        return float(headers.get('x-tts-temperature', '0.8'))

    def _extract_tts_cfg_weight(headers):
        return float(headers.get('x-tts-cfg-weight', '0.5'))
    
    def _extract_model(model: str):
        return model.removeprefix('replicate/')

    def run_tts(self, model: str, text: str, voice: str = None, headers: Optional[Dict[str, str]] = None, voice_info=None, **kwargs):
        if not voice_info:
            voice_info = {}
            
        language = voice_info.get('language', 'en')
            
        response = replicate.client.Client(api_token=self._extract_api_key(headers)).run(
            self._extract_model(model),
            input={
                "text": text,
                "reference_audio": open(voice or voice_info.get('path'), 'rb'),
                "cfg_weight": self._extract_tts_cfg_weight if language in ['pt', 'en'] else 0.0,
                "language_id": language,
                "exaggeration": self._extract_tts_exaggeration(headers),
                "temperature": self._extract_tts_temperature(headers),
            }
        )
        
        return response.read()
        
        