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

    def _extract_api_key(self, headers: Optional[Dict[str, Any]]) -> Optional[str]:
        if not headers:
            return None
        return headers.get("x-api-key", None)

    def _extract_base_url(self, headers: Optional[Dict[str, str]]) -> Optional[str]:
        if not headers:
            return None
        return headers.get("x-base-url", None)

    async def run_chat(self, model: str, messages: List[Dict[str, Any]], stream: bool = False, headers: Optional[Dict[str, str]] = None, **kwargs):
        """Runs chat/vision completions via litellm."""
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        api_base = self._extract_base_url(headers)
        if api_base:
            params["api_base"] = api_base
            
        api_key = self._extract_api_key(headers)
        if api_key:
            params["api_key"] = api_key
        
        try:
            from litellm import acompletion
            return await acompletion(**params)
        except Exception as e:
            logger.error(f"LLMApi: Chat failed: {e}")
            raise e

    def run_embeddings(self, model: str, input_text: Union[str, List[str]], headers: Optional[Dict[str, str]] = None, **kwargs):
        """Runs embeddings via litellm."""
        params = {
            "model": model,
            "input": [input_text] if isinstance(input_text, str) else input_text,
            **kwargs
        }
        
        api_base = self._extract_base_url(headers)
        if api_base:
            params["api_base"] = api_base
            
        api_key = self._extract_api_key(headers)
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = embedding(**params)
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error(f"LLMApi: Embeddings failed: {e}")
            raise e

    def run_tts(self, model: str, text: str, voice: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> bytes:
        """Runs text-to-speech via litellm (OpenAI compatible)."""
        params = {
            "model": model,
            "input": text,
            "voice": voice,
            **kwargs
        }
        
        api_base = self._extract_base_url(headers)
        if api_base:
            params["api_base"] = api_base
            
        api_key = self._extract_api_key(headers)
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = speech(**params)
            return response.content
        except Exception as e:
            logger.error(f"LLMApi: TTS failed: {e}")
            raise e

    def run_stt(self, model: str, audio_file: Any, headers: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Runs speech-to-text via litellm."""
        params = {
            "model": model,
            "file": audio_file,
            **kwargs
        }
        
        api_base = self._extract_base_url(headers)
        if api_base:
            params["api_base"] = api_base
            
        api_key = self._extract_api_key(headers)
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

    def run_image_gen(self, model: str, prompt: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Runs image generation via litellm. Returns base64 string."""
        params = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        
        api_base = self._extract_base_url(headers)
        if api_base:
            params["api_base"] = api_base
            
        api_key = self._extract_api_key(headers)
        if api_key:
            params["api_key"] = api_key
            
        try:
            response = image_generation(**params)
            raw_data = response.data[0].get("b64_json") or response.data[0].get("url")
            return self._ensure_base64(raw_data)
        except Exception as e:
            logger.error(f"LLMApi: Image generation failed: {e}")
            raise e

    def run_image_edit(self, model: str, prompt: str, image: Any, headers: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Runs image editing directly via requests since litellm lacks image_editing."""
        api_base = self._extract_base_url(headers) or "https://api.openai.com/v1"
        api_key = self._extract_api_key(headers)
        
        req_headers = {}
        if api_key:
            req_headers["Authorization"] = f"Bearer {api_key}"
            
        url = f"{api_base.rstrip('/')}/images/edits"
        
        files = {
            "image": ("image.png", image, "image/png")
        }
        
        data = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        
        try:
            import requests
            response = requests.post(url, headers=req_headers, data=data, files=files)
            response.raise_for_status()
            res_json = response.json()
            raw_data = res_json["data"][0].get("b64_json") or res_json["data"][0].get("url")
            return self._ensure_base64(raw_data)
        except Exception as e:
            logger.error(f"LLMApi: Image edit failed: {e}")
            raise e
