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
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"DownloadImage [POST]: Tentativa {attempt}/{retries} falhou na API {url}. Erro: {str(e.response.text)}")
                
                if e.response.status_code in [400, 422]:
                    raise
                
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

    _TASK_DISPATCH: dict = {}  # populated after method definitions

    def __init__(self, backend="api", model=None):
        self.backend = backend
        self.model = model or {}

    async def dispatch(self, task_type: str, payload: dict):
        """Entry point called by Bridge for api backend. Routes by task_type."""
        handler = self._TASK_DISPATCH.get(task_type)
        if not handler:
            raise ValueError(f"LLMApi: unsupported task_type '{task_type}'")
        return await handler(self, payload)

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """API baseada em nuvem não consome VRAM local."""
        return 0

    def _extract_api_key(self, headers: Optional[Dict[str, Any]]) -> Optional[str]:
        config_key = self.model.get("api_key", None)
        
        if not headers:
            return config_key

        return headers.get("x-api-key", config_key)

    def _extract_base_url(self, headers: Optional[Dict[str, str]]) -> Optional[str]:
        config_url = self.model.get("base_url", None)
        if not headers:
            return config_url
        return headers.get("x-base-url", self.model.get("base_url", None))

    async def run_chat(self, payload: dict, **kwargs):
        """Runs chat/vision completions via litellm."""
        import litellm
        from litellm import acompletion
        # litellm.drop_params = True

        headers = payload.pop("headers", {})
        payload_model = payload.pop("model", None)
        model = self.model.get("model", None) or payload_model
        reasoning_enabled = payload.pop("reasoning_enabled", None) or kwargs.pop("reasoning_enabled", None) or self.model.get("reasoning_enabled", False)
        reasoning_effort = payload.pop("reasoning_effort", None) or kwargs.pop("reasoning_effort", None) or self.model.get("reasoning_effort", "medium")
        
        params = {
            "model": model,
            **kwargs,
            **payload
        }

        # Enable usage reporting in stream if requested
        if params.get("stream") is True:
            if "stream_options" not in params:
                params["stream_options"] = {"include_usage": True}
            else:
                params["stream_options"]["include_usage"] = True
        
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
        """Correção para suportar o padrão de array de imagens (image[])"""
        api_base = self._extract_base_url(headers) or "https://api.openai.com/v1"
        api_key = self._extract_api_key(headers)
        
        req_headers = {}
        if api_key:
            req_headers["Authorization"] = f"Bearer {api_key}"
            
        url = f"{api_base.rstrip('/')}/images/edits"
        
        # 1. Construindo a lista de arquivos no formato array (image[])
        # No httpx, para repetir a mesma chave, usamos uma lista de tuplas
        file_list = []
        
        # Adiciona a imagem principal (ou a primeira da lista)
        file_list.append(("image[]", ("image_1.png", image, "image/png")))
        file_list.append(("image", ("image_1.png", image, "image/png")))
        
        # Se 'image' for na verdade uma lista de imagens (baseado no seu curl)
        # Você poderia iterar aqui. Mas se o parâmetro 'image' for um só:
        if mask:
            file_list.append(("image[]", ("mask.png", mask, "image/png")))

        # 2. Dados de Texto
        data = {
            "prompt": prompt,
            "model": model
        }
        
        # Sanitiza kwargs
        for k, v in kwargs.items():
            if v is not None:
                data[k] = str(v)
        
        try:
            # Note que passamos file_list (lista de tuplas) para files
            res_json = await DownloadImage.post_api_with_retry(
                url, 
                headers=req_headers, 
                data=data, 
                files=file_list, 
                retries=10
            )
            
            # O retorno costuma seguir o padrão OpenAI
            raw_data = res_json["data"][0].get("b64_json") or res_json["data"][0].get("url")
            logger.info(f"Engine LLMApi (ImageEdit) executado com sucesso: {model}")
            return self._ensure_base64(raw_data)
                
        except HTTPStatusError as e:
            logger.error(f"Erro na API DeepInfra: {e.response.status_code} - {e.response.text}")
            raise e
        except Exception as e:
            logger.error(f"Erro fatal no image edit: {str(e)}")
            raise e

    # ------------------------------------------------------------------
    # Payload-aware private handlers used by dispatch()
    # ------------------------------------------------------------------

    async def _handle_chat(self, payload: dict):
        return await self.run_chat(payload)

    async def _handle_embeddings(self, payload: dict):
        headers = payload.get("headers", {})
        model = payload.get("model") or self.model.get("model", "")
        input_text = payload.get("input") or payload.get("text", "")
        return await self.run_embeddings(model, input_text, headers)

    async def _handle_tts(self, payload: dict):
        headers = payload.get("headers", {})
        model = payload.get("model") or self.model.get("model", "")
        text = payload.get("input") or payload.get("text", "")
        voice = payload.get("voice", "alloy")
        return await self.run_tts(model, text, voice, headers)

    async def _handle_stt(self, payload: dict):
        headers = payload.get("headers", {})
        model = payload.get("model") or self.model.get("model", "")
        audio_file = payload.get("audio") or payload.get("file")
        language = payload.get("language")
        return await self.run_stt(model, audio_file, headers, language=language)

    async def _handle_image_generation(self, payload: dict):
        headers = payload.get("headers", {})
        model = payload.get("model") or self.model.get("model", "")
        prompt = payload.get("prompt", "")
        return await self.run_image_gen(model, prompt, headers)

    async def _handle_image_edit(self, payload: dict):
        headers = payload.get("headers", {})
        model = payload.get("model") or self.model.get("model", "")
        prompt = payload.get("prompt", "")
        image = payload.get("image")
        mask = payload.get("mask")
        return await self.run_image_edit(model, prompt, image, mask, headers)

    async def _handle_web_search(self, payload: dict):
        query = payload.get("query", "")
        headers = payload.get("headers", {})
        engine = (payload.get("model") or "").lower()
        if engine == "google":
            from ..models.google_search import GoogleSearchModel
            return await GoogleSearchModel().search(query, headers=headers)
        from ..models.tavily_search import TavilySearchModel
        return await TavilySearchModel().search(
            query,
            headers=headers,
            search_depth=payload.get("search_depth", "basic"),
            include_answer=payload.get("include_answer", True),
            max_results=payload.get("max_results", 5),
            include_list=payload.get("include_list", False),
        )

    async def _handle_code_search(self, payload: dict):
        query = payload.get("query", "")
        from ..models.github_search import GithubSearchModel
        return await GithubSearchModel().search(query)


# Registry of task_type → unbound handler method.
# Dictionary lookup replaces any if/elif chain.
LLMApi._TASK_DISPATCH = {
    "text_chat":        LLMApi._handle_chat,
    "vision_chat":      LLMApi._handle_chat,
    "embeddings":       LLMApi._handle_embeddings,
    "tts":              LLMApi._handle_tts,
    "stt":              LLMApi._handle_stt,
    "image_generation": LLMApi._handle_image_generation,
    "image_edit":       LLMApi._handle_image_edit,
    "web_search":       LLMApi._handle_web_search,
    "code_search":      LLMApi._handle_code_search,
}

llm_api = LLMApi()
