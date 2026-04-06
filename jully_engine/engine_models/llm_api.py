from __future__ import annotations
import os
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from httpx import HTTPStatusError

if TYPE_CHECKING:
    import litellm
    from litellm import completion, embedding, image_generation, transcription, speech, acompletion, aembedding, aimage_generation, atranscription, aspeech

logger = logging.getLogger("JulyEngine.Models.LLMApi")


class DownloadImage:
    """
    Classe utilitária para garantir resiliência em requisições de rede severas
    com retry automático caso o servidor (Cloudflare/Gateway) aborte a conexão.
    """
    @staticmethod
    async def post_api_with_retry(url: str, headers: dict, data: dict, files: dict, retries: int = 10) -> dict:
        import httpx
        last_error = None
        
        for attempt in range(1, retries + 1):
            try:
                # timeout=None garante que o nosso lado não corte a conexão prematuramente
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.post(url, headers=headers, data=data, files=files)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                last_error = e
                logger.warning(f"DownloadImage [POST]: Tentativa {attempt}/{retries} falhou na API {url}. Erro: {str(e)}")
                if attempt < retries:
                    await asyncio.sleep(2)  # Backoff simples antes da próxima tentativa
        
        logger.error(f"DownloadImage [POST]: Falha fatal após {retries} tentativas.")
        raise last_error

    @staticmethod
    def get_base64_sync_with_retry(url: str, retries: int = 10) -> str:
        import requests
        import base64
        import time
        last_error = None
        
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=None)
                response.raise_for_status()
                return base64.b64encode(response.content).decode("utf-8")
            except Exception as e:
                last_error = e
                logger.warning(f"DownloadImage [GET]: Tentativa {attempt}/{retries} falhou ao baixar arquivo. Erro: {str(e)}")
                if attempt < retries:
                    time.sleep(2)
                    
        logger.error(f"DownloadImage [GET]: Falha fatal de download após {retries} tentativas.")
        raise last_error


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
        import litellm
        from litellm import acompletion
        litellm.drop_params = True

        reasoning_enabled = kwargs.pop("reasoning_enabled", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if reasoning_enabled:
            params["extra_body"] = {
                "reasoning": {
                    "enabled": True,
                    "effort": reasoning_effort or "medium"
                }
            }
        
        api_base = self._extract_base_url(headers)
        if api_base:
            params["api_base"] = api_base
            
        api_key = self._extract_api_key(headers)
        if api_key:
            params["api_key"] = api_key
        
        try:
            res = await acompletion(**params)
            logger.info(f"Engine LLMApi (Chat) executed successfully on {self.backend} with {model}")
            return res
        except Exception as e:
            logger.error(f"LLMApi: Chat failed: {e}")
            raise e

    async def run_embeddings(self, model: str, input_text: Union[str, List[str]], headers: Optional[Dict[str, str]] = None, **kwargs):
        """Runs embeddings via litellm."""
        import litellm
        from litellm import aembedding
        litellm.drop_params = True

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
            response = await aembedding(**params)
            logger.info(f"Engine LLMApi (Embeddings) executed successfully on {self.backend} with {model}")
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error(f"LLMApi: Embeddings failed: {e}")
            raise e

    async def run_tts(self, model: str, text: str, voice: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> bytes:
        """Runs text-to-speech via litellm (OpenAI compatible)."""
        import litellm
        from litellm import aspeech
        litellm.drop_params = True

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
            response = await aspeech(**params)
            logger.info(f"Engine LLMApi (TTS) executed successfully on {self.backend} with {model}")
            return response.content
        except Exception as e:
            logger.error(f"LLMApi: TTS failed: {e}")
            raise e

    async def run_stt(self, model: str, audio_file: Any, headers: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Runs speech-to-text via litellm."""
        import litellm
        from litellm import atranscription
        litellm.drop_params = True

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
            response = await atranscription(**params)
            logger.info(f"Engine LLMApi (STT) executed successfully on {self.backend} with {model}")
            return response.text
        except Exception as e:
            logger.error(f"LLMApi: STT failed: {e}")
            raise e

    def _ensure_base64(self, data: str) -> str:
        """Helper to ensure image data is returned as base64 string using DownloadImage retries."""
        if not data:
            return ""
        if data.startswith("http"):
            logger.info(f"LLMApi: Downloading image from {data} (with 10 retries)")
            return DownloadImage.get_base64_sync_with_retry(data, retries=10)
        return data

    async def run_image_gen(self, model: str, prompt: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Runs image generation via litellm. Returns base64 string."""
        import litellm
        from litellm import aimage_generation
        litellm.drop_params = True

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
            response = await aimage_generation(**params)
            raw_data = response.data[0].get("b64_json") or response.data[0].get("url")
            logger.info(f"Engine LLMApi (ImageGen) executed successfully on {self.backend} with {model}")
            return self._ensure_base64(raw_data)
        except Exception as e:
            logger.error(f"LLMApi: Image generation failed: {e}")
            raise e

    async def run_image_edit(self, model: str, prompt: str, image: Any, mask: Optional[Any] = None, headers: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Runs image editing directly via httpx since litellm lacks image_editing, now wrapped in DownloadImage."""
        api_base = self._extract_base_url(headers) or "https://api.openai.com/v1"
        api_key = self._extract_api_key(headers)
        
        req_headers = {}
        
        if api_key:
            req_headers["Authorization"] = f"Bearer {api_key}"
            
        url = f"{api_base.rstrip('/')}/images/edits"
        
        # 1. Arquivos Binários (Obrigatório para o endpoint)
        files = {
            "image": ("image.png", image, "image/png")
        }
        
        # Acopla a máscara como arquivo se ela existir
        if mask:
            files["mask"] = ("mask.png", mask, "image/png")
        
        # 2. Dados de Texto (httpx exige que todos os valores sejam strings)
        data = {
            "prompt": prompt,
            "model": model
        }
        
        # Sanitiza os kwargs (ex: n=1 vira n="1")
        for k, v in kwargs.items():
            if v is not None:
                data[k] = str(v)
        
        try:
            res_json = await DownloadImage.post_api_with_retry(url, headers=req_headers, data=data, files=files, retries=10)
            raw_data = res_json["data"][0].get("b64_json") or res_json["data"][0].get("url")
            logger.info(f"Engine LLMApi (ImageEdit) executed successfully on {self.backend} with {model}")
            return self._ensure_base64(raw_data)
                
        except HTTPStatusError as e:
            logger.error(f"LLMApi: Image edit failed: {e.response.status_code} - {e.response.content.decode('utf-8')}")
            raise e
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"LLMApi: FATAL error in image edit:\n{error_trace}")
            raise e