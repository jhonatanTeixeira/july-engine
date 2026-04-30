from __future__ import annotations
import os
from typing import Dict, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import replicate
    import replicate.client

logger = logging.getLogger("JulyEngine.Models.Replicate")

class Replicate:
    def __init__(self, backend="api"):
        self.backend = backend

    def is_loaded(self):
        """Replicate é baseado em API, sempre 'carregado'."""
        return True

    def _extract_api_key(self, headers):
        if not headers: return None
        return headers.get('x-api-key')
    
    def _extract_tts_exaggeration(self, headers):
        if not headers: return 0.5
        return float(headers.get('x-tts-exageration', '0.5'))
    
    def _extract_tts_temperature(self, headers):
        if not headers: return 0.8
        return float(headers.get('x-tts-temperature', '0.8'))

    def _extract_tts_cfg_weight(self, headers):
        if not headers: return 0.5
        return float(headers.get('x-tts-cfg-weight', '0.5'))
    
    def _extract_model(self, model: str):
        return model.removeprefix('replicate/')

    async def run_tts(self, model: str, text: str, voice: str = None, headers: Optional[Dict[str, str]] = None, **kwargs):
        model = self._extract_model(model)

        from ..services.voice_service import voice_service
        
        voice_res = voice_service.get_voice_path(voice)

        if not voice_res:
            logger.error(f"Replicate: Voice {voice} not found and no fallback available.")
            raise ValueError(f"Voice {voice} not found")
            
        full_path, voice_lang = voice_res
        language = voice_lang or 'en'
        
        if not os.path.exists(full_path):
            logger.error(f"Replicate: Reference audio not found at {full_path} for voice ID '{voice}'")
            raise FileNotFoundError(f'Audio path does not exist: {full_path}')

        try:
            with open(full_path, 'rb') as audio_file:
                if model.find('chatterbox') > 0:
                    input_data = {
                        "text": text,
                        "cfg_weight": self._extract_tts_cfg_weight(headers) if language in ['pt', 'en'] else 0.0,
                        "language": language,
                        "exaggeration": self._extract_tts_exaggeration(headers),
                        "temperature": self._extract_tts_temperature(headers),
                        "reference_audio": audio_file
                    }
                elif model.find('xtts-v2') > 0:
                    input_data = {
                        'text': text,
                        'speaker': audio_file,
                        'language': language
                    }
                elif model.find('qwen3_tts') > 0:
                    input_data = {
                        'mode': 'voice_clone',
                        'text': text,
                        'reference_audio': audio_file,
                        'language': language
                    }

                    if (style := kwargs.get('style_instruction', None)):
                        input_data['style_instruction'] = style
                        
                else:
                    input_data = {
                        'text': text,
                        **kwargs
                    }

                log_data = input_data.copy()
                if "reference_audio" in log_data:
                    log_data["reference_audio"] = f"<File pointer to {full_path}>"
                if "speaker" in log_data:
                    log_data["speaker"] = f"<File pointer to {full_path}>"
                
                logger.info(f"Replicate TTS Payload: {log_data}")

                import asyncio
                import replicate
                import replicate.client
                client = replicate.client.Client(api_token=self._extract_api_key(headers))
                
                def _do_run():
                    response = client.run(model, input=input_data)
                    return response.read()

                audio_content = await asyncio.to_thread(_do_run)

                logger.info(f"Engine Replicate executed successfully on {self.backend} with {model}")
                return audio_content
                
        except Exception as e:
            logger.error(f"Replicate: TTS failed: {e}")
            raise e