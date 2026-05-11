import logging
import json
import copy
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

logger = logging.getLogger("JulyEngine.Bridge")


class Bridge:
    """
    Dumb routing middleware.

    Responsibilities:
      1. Inject HTTP headers into payload.
      2. Resolve model backend (gpu / cpu / api) from model config.
      3. Route to the right orchestrator or llm_api.dispatch().

    No domain knowledge, no data conversion.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Inicializa os orquestradores de GPU e CPU."""
        from .orchestrator import gpu_orchestrator, cpu_orchestrator
        await gpu_orchestrator.start()
        await cpu_orchestrator.start()

    async def stop(self):
        """Finaliza os orquestradores e libera recursos."""
        from .orchestrator import gpu_orchestrator, cpu_orchestrator
        await gpu_orchestrator.stop()
        await cpu_orchestrator.stop()

    # ------------------------------------------------------------------
    # Routing Helpers
    # ------------------------------------------------------------------

    _TASK_TO_SETTING = {
        "text_chat": "TEXT_PRESETS",
        "vision_chat": "VISION",
        "tts": "TTS",
        "stt": "STT",
        "embeddings": "EMBEDDINGS",
        "image_generation": "IMAGE_CREATE",
        "image_edit": "IMAGE_EDIT",
        "image_resize": "RESIZE",
        "image_remove_background": "BG_REMOVAL",
        "web_search": "WEB_SEARCH",
        "code_search": "REPOSITORY_SEARCH",
        "video_description": "VISION"
    }

    def _inject_headers(self, payload: dict, headers: dict) -> dict:
        payload["headers"] = headers
        return payload

    def _get_backend(self, payload: dict) -> str:
        """Determina o backend baseado no payload e configurações."""

        backend = payload.get("headers", {}).get("x-backend", None)

        if not backend:
            from .services.models_service import model_service
            model = model_service.resolve_by_settings(payload.get("model"))

            backend = model.get("backend", "api")

        return backend
            

    async def _dispatch(self, task_type: str, payload: dict, headers: dict):
        self._inject_headers(payload, headers)
        
        backend = self._get_backend(payload)

        if backend == "gpu":
            from .orchestrator import gpu_orchestrator
            return await gpu_orchestrator.submit_task(task_type, payload)
        if backend == "cpu":
            from .orchestrator import cpu_orchestrator
            return await cpu_orchestrator.submit_task(task_type, payload)

        from .services.llm_api import llm_api
        return await llm_api.dispatch(task_type, payload)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def process_openai_chat(self, payload: dict, headers: dict):
        return await self._dispatch('text_chat', payload, headers)

    async def process_anthropic_message(self, payload: dict, headers: dict):
        # Mapeia o prompt de sistema do Anthropic para uma mensagem de sistema no estilo OpenAI
        system_prompt = payload.pop("system", None)
        
        if system_prompt:
            messages = payload.get("messages", [])
            # Insere no início se ainda não houver sistema
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
            payload["messages"] = messages

        response = await self._dispatch("text_chat", payload, headers)
        
        if isinstance(response, AsyncGenerator):
            return response
            
        # Converter formato OpenAI para Anthropic se necessário
        if isinstance(response, dict) and "choices" in response:
            msg = response["choices"][0].get("message", {})
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return {
                "id": response.get("id", ""),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "model": response.get("model", ""),
                "usage": {
                    "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": response.get("usage", {}).get("completion_tokens", 0)
                }
            }
        return response

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    async def process_tts(self, payload: dict, headers: dict):
        return await self._dispatch("tts", payload, headers)

    async def process_stt(self, payload: dict, headers: dict):
        return await self._dispatch("stt", payload, headers)

    # ------------------------------------------------------------------
    # Image Generation / Editing
    # ------------------------------------------------------------------

    async def process_image_generation(self, payload: dict, headers: dict):
        return await self._dispatch("image_generation", payload, headers)

    async def process_image_edit(self, payload: dict, headers: dict):
        return await self._dispatch("image_edit", payload, headers)

    async def process_image_resize(self, payload: dict, headers: dict):
        return await self._dispatch("image_resize", payload, headers)

    async def process_image_remove_background(self, payload: dict, headers: dict):
        return await self._dispatch("image_remove_background", payload, headers)

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    async def process_image_description(self, payload: dict, headers: dict):
        return await self._dispatch("vision_chat", payload, headers)

    async def process_video_description(self, payload: dict, headers: dict):
        from .services.video_processing import video_processing_service
        
        video_path = payload.get("video_path")
        interval_sec = float(payload.get("interval_sec", 2.0))
        frames_per_grid = int(payload.get("frames_per_grid", 1))
        strategy = payload.get("strategy", "default")
        detect_changes = payload.get("detect_changes", False)
        
        return await video_processing_service.execute(
            video_path, 
            interval_sec=interval_sec, 
            frames_per_grid=frames_per_grid, 
            strategy=strategy, 
            detect_changes=detect_changes, 
            headers=headers
        )

    async def process_face_extraction(self, payload: dict, headers: dict):
        return await self._dispatch("face_extraction", payload, headers)

    async def process_face_sync_batch(self, payload: dict, headers: dict):
        return await self._dispatch("face_sync", payload, headers)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def process_embeddings(self, payload: dict, headers: dict):
        return await self._dispatch("embeddings", payload, headers)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def process_web_search(self, payload: dict, headers: dict):
        return await self._dispatch("web_search", payload, headers)

    async def process_code_search(self, payload: dict, headers: dict):
        return await self._dispatch("code_search", payload, headers)

    async def process_search_web(self, payload: dict, headers: dict):
        """Alias para process_web_search usado pelos roteadores."""
        return await self.process_web_search(payload, headers)

    async def process_search_code(self, payload: dict, headers: dict):
        """Alias para process_code_search usado pelos roteadores."""
        return await self.process_code_search(payload, headers)

    # ------------------------------------------------------------------
    # RAG (Memory)
    # ------------------------------------------------------------------

    async def process_rag_add(self, payload: dict, headers: dict):
        return await self._dispatch("rag_add", payload, headers)

    async def process_rag_batch_add(self, payload: dict, headers: dict):
        return await self._dispatch("rag_batch_add", payload, headers)

    async def process_rag_search(self, payload: dict, headers: dict):
        return await self._dispatch("rag_search", payload, headers)

    async def process_rag_vector_add(self, payload: dict, headers: dict):
        return await self._dispatch("rag_vector_add", payload, headers)

    async def process_rag_update(self, payload: dict, headers: dict):
        return await self._dispatch("rag_update", payload, headers)

    async def process_rag_delete(self, payload: dict, headers: dict):
        return await self._dispatch("rag_delete", payload, headers)

    async def process_rag_list(self, payload: dict, headers: dict):
        return await self._dispatch("rag_list", payload, headers)

    async def process_rag_smart_search(self, payload: dict, headers: dict):
        return await self._dispatch("rag_smart_search", payload, headers)

    async def process_search_and_scrape(self, results: list, query: str, headers: dict, describe_model: str = None):
        """Raspagem de URLs e sumarização via LLM."""
        from .services.scraper_service import scraper_service
        
        # Extrai URLs (limitado a 3 para performance)
        urls = [r["url"] for r in results if isinstance(r, dict) and "url" in r][:3]
        if not urls:
            return "Nenhuma URL encontrada para raspagem."
            
        scraped = await scraper_service.scrape_urls(urls)
        
        # Combina o conteúdo
        context = "\n\n".join([f"Fonte: {s['url']}\nConteúdo: {s['content']}" for s in scraped])
        
        # Prompt de sumarização
        prompt = f"Com base no contexto abaixo, responda à pergunta do usuário: {query}\n\nContexto:\n{context}"
        
        summarize_payload = {
            "model": describe_model or "default",
            "messages": [
                {"role": "system", "content": "Você é um assistente de pesquisa. Resuma os resultados de forma concisa em português."},
                {"role": "user", "content": prompt}
            ]
        }
        
        return await self.process_openai_chat(summarize_payload, headers)


bridge = Bridge()
