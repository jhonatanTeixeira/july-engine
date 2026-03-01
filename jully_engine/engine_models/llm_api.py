import os
import logging
from typing import Any, Dict, List, Optional, Union
import litellm
from litellm import completion, embedding, image_generation, transcription, speech

logger = logging.getLogger("JulyEngine.Models.LLMApi")

class LLMApi:
    """
    Unified API strategy using litellm for all domains.
    Supported: Chat, Vision, Embeddings, TTS, STT, Image Gen/Edit.
    """
    def __init__(self, backend="api"):
        self.backend = backend

    def _extract_api_key(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Extracts Bearer token from headers if present."""
        headers = kwargs.get("headers", {})
        auth = headers.get("Authorization") or headers.get("authorization")
        if auth and isinstance(auth, str) and auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def run_chat(self, model: str, messages: List[Dict[str, Any]], stream: bool = False, base_url: Optional[str] = None, **kwargs):
        """Runs chat/vision completions via litellm."""
        api_key = self._extract_api_key(kwargs)
        
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if base_url:
            params["api_base"] = base_url
        if api_key:
            params["api_key"] = api_key
            
        try:
            return completion(**params)
        except Exception as e:
            logger.error(f"LLMApi: Chat failed: {e}")
            raise e

    def run_embeddings(self, model: str, input_text: Union[str, List[str]], base_url: Optional[str] = None, **kwargs):
        """Runs embeddings via litellm."""
        api_key = self._extract_api_key(kwargs)
        params = {
            "model": model,
            "input": [input_text] if isinstance(input_text, str) else input_text,
            **kwargs
        }
        if base_url:
            params["api_base"] = base_url
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = embedding(**params)
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error(f"LLMApi: Embeddings failed: {e}")
            raise e

    def run_tts(self, model: str, text: str, voice: str, base_url: Optional[str] = None, **kwargs) -> bytes:
        """Runs text-to-speech via litellm (OpenAI compatible)."""
        api_key = self._extract_api_key(kwargs)
        params = {
            "model": model,
            "input": text,
            "voice": voice,
            **kwargs
        }
        if base_url:
            params["api_base"] = base_url
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = speech(**params)
            return response.content
        except Exception as e:
            logger.error(f"LLMApi: TTS failed: {e}")
            raise e

    def run_stt(self, model: str, audio_file: Any, base_url: Optional[str] = None, **kwargs) -> str:
        """Runs speech-to-text via litellm."""
        api_key = self._extract_api_key(kwargs)
        params = {
            "model": model,
            "file": audio_file,
            **kwargs
        }
        if base_url:
            params["api_base"] = base_url
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = transcription(**params)
            return response.text
        except Exception as e:
            logger.error(f"LLMApi: STT failed: {e}")
            raise e

    def _ensure_base64(self, data: str) -> str:
        """Helper to ensure image data is returned as base64 string."""
        if not data:
            return ""
        if data.startswith("http"):
            import requests
            import base64
            try:
                logger.info(f"LLMApi: Downloading image from {data}")
                response = requests.get(data)
                response.raise_for_status()
                return base64.b64encode(response.content).decode("utf-8")
            except Exception as e:
                logger.error(f"LLMApi: Failed to download image from URL: {e}")
                raise e
        return data

    def run_image_gen(self, model: str, prompt: str, base_url: Optional[str] = None, **kwargs) -> str:
        """Runs image generation via litellm. Returns base64 string."""
        api_key = self._extract_api_key(kwargs)
        params = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        if base_url:
            params["api_base"] = base_url
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = image_generation(**params)
            # Returns first image data (url or b64_json)
            raw_data = response.data[0].get("b64_json") or response.data[0].get("url")
            return self._ensure_base64(raw_data)
        except Exception as e:
            logger.error(f"LLMApi: Image generation failed: {e}")
            raise e

    def run_image_edit(self, model: str, prompt: str, image: Any, base_url: Optional[str] = None, **kwargs) -> str:
        """Runs image editing via litellm. Returns base64 string."""
        api_key = self._extract_api_key(kwargs)
        params = {
            "model": model,
            "prompt": prompt,
            "image": image,
            **kwargs
        }
        if base_url:
            params["api_base"] = base_url
        if api_key:
            params["api_key"] = api_key
            
        try:
            # Note: litellm image_editing support varies
            import litellm
            response = litellm.image_editing(**params)
            raw_data = response.data[0].get("b64_json") or response.data[0].get("url")
            return self._ensure_base64(raw_data)
        except Exception as e:
            logger.error(f"LLMApi: Image edit failed: {e}")
            raise e
