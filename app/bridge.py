import logging
import json
import copy
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from july_routers.bridge_interface import BridgeInterface

logger = logging.getLogger("JulyEngine.Bridge")


class Bridge(BridgeInterface):
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
        from .orchestrator import orchestrator
        await orchestrator.start()

    async def stop(self):
        """Finaliza os orquestradores e libera recursos."""
        from .orchestrator import orchestrator
        await orchestrator.stop()

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
        "video_description": "VISION",
        "video_generation": "VIDEO_GENERATION",
    }

    def _inject_headers(self, payload: dict, headers: dict) -> dict:
        payload["headers"] = headers
        return payload

    async def _dispatch(self, task_type: str, payload: dict, headers: dict):
        from .orchestrator import orchestrator

        self._inject_headers(payload, headers)
        
        return await orchestrator.submit_task(task_type, payload)

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

    async def process_pdf_extract(self, pdf_bytes: bytes):
        from .services.pdf_extractor import extract_pdf
        return extract_pdf(pdf_bytes)

    async def process_resource_check(self, payload: dict):
        from llama_gguf.resource_calculator import estimate_vram_ram
        return await estimate_vram_ram(
            model_path=payload.get("model_path", "model"),
            context_window=payload.get("context_window", "4k"),
            kv_cache_quantization=payload.get("kv_cache_quantization", "FP16"),
            gpu_layers=payload.get("gpu_layers") if payload.get("gpu_layers") != -1 else None,
            repo_id=payload.get("model_id"),
            filename=payload.get("filename"),
            mmproj_path=payload.get("mmproj_path"),
            mmproj_repo_id=payload.get("mmproj_id"),
            mmproj_filename=payload.get("mmproj_filename"),
            n_seq_max=payload.get("n_seq_max", 1),
            offload_kqv=payload.get("offload_kqv", True),
            flash_attention=payload.get("flash_attn", True),
            logits_all=payload.get("logits_all", False),
            vision_on_cpu=payload.get("vision_on_cpu", False),
        )

    async def process_video_generation(self, payload: dict, headers: dict):
        return await self._dispatch("video_generation", payload, headers)

    def _extract_llm_text(self, response) -> str:
        """Extrai o texto de content de uma resposta OpenAI Chat Completion."""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0].get("message", {}).get("content", "") or ""
        return str(response)

    async def process_search_and_scrape(self, results: list, query: str, headers: dict, describe_model: str = None):
        """Raspagem de URLs (via Tavily results) e sumarização via LLM."""
        urls = [r["url"] for r in results if isinstance(r, dict) and "url" in r][:3]
        if not urls:
            return "Nenhuma URL encontrada para raspagem."
        return await self.process_scrape_and_summarize(urls, query, headers, model=describe_model)

    async def process_scrape_and_summarize(
        self,
        urls: list,
        query: str,
        headers: dict,
        context_window: int = None,
        model: str = None,
    ) -> str:
        """
        Raspa todas as URLs fornecidas e sumariza via LLM com suporte a chunking.

        Se o conteúdo total ultrapassar a janela de contexto configurada do modelo,
        divide em chunks, sumariza cada um e combina os resumos parciais.
        """
        from .services.scraper_service import scraper_service

        if not urls:
            return "Nenhuma URL fornecida para raspagem."

        # Descobre a janela de contexto do modelo se não fornecida
        if not context_window:
            try:
                from .services.models_service import model_service
                presets = model_service.get_setting("TEXT_PRESETS") or []
                if isinstance(presets, list) and presets:
                    preset = presets[0]
                    model_alias = preset.get("alias") or preset.get("model")
                    model_meta = model_service.get(model_alias) if model_alias else None
                    if model_meta and model_meta.get("context_window"):
                        context_window = int(model_meta["context_window"])
            except Exception:
                pass
            context_window = context_window or 8192

        # ~4 chars/token; reserva 30% para prompt e output
        max_chars_per_chunk = int(context_window * 4 * 0.65)

        logger.info(f"Scraping {len(urls)} URLs (context_window={context_window}, max_chars/chunk={max_chars_per_chunk})")

        scraped = await scraper_service.scrape_urls(urls)

        pages = [
            f"Fonte: {s['url']}\nConteúdo: {s['content']}"
            for s in scraped
            if s.get("content") and not s["content"].startswith("Error:")
        ]

        if not pages:
            return "Nenhum conteúdo pôde ser raspado das URLs fornecidas."

        # Divide em chunks que cabem na janela de contexto
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0
        for page in pages:
            if current_len + len(page) > max_chars_per_chunk and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = [page]
                current_len = len(page)
            else:
                current_parts.append(page)
                current_len += len(page)
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        logger.info(f"Conteúdo dividido em {len(chunks)} chunk(s)")

        chat_headers = {k: v for k, v in headers.items() if k.lower() != "x-backend"}
        llm_model = model or "default"

        async def summarize_chunk(i: int, chunk: str) -> str:
            label = f"chunk {i+1}/{len(chunks)}" if len(chunks) > 1 else "conteúdo"
            payload = {
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente de pesquisa especializado em síntese técnica. "
                            "Resuma o conteúdo abaixo em português, preservando todos os fatos, "
                            "dados, datas, estatísticas e referências importantes. "
                            "Seja detalhado e abrangente."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Pesquisa: {query}\n\n"
                            f"Conteúdo ({label}):\n{chunk}"
                        ),
                    },
                ],
            }
            response = await self.process_openai_chat(payload, chat_headers)
            return self._extract_llm_text(response)

        import asyncio
        summaries = await asyncio.gather(*[summarize_chunk(i, c) for i, c in enumerate(chunks)])
        summaries = [s for s in summaries if s.strip()]

        if len(summaries) == 1:
            return summaries[0]

        # Combina os resumos parciais em um resumo final
        combined = "\n\n---\n\n".join(summaries)
        if len(combined) <= max_chars_per_chunk:
            final_payload = {
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Você é um pesquisador sênior. Consolide os resumos parciais abaixo "
                            "em um único documento coerente, detalhado e bem estruturado em português. "
                            "Elimine repetições, mas preserve todos os fatos distintos."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Pesquisa: {query}\n\nResumos parciais:\n{combined}",
                    },
                ],
            }
            response = await self.process_openai_chat(final_payload, chat_headers)
            return self._extract_llm_text(response) or combined

        return combined


bridge = Bridge()
