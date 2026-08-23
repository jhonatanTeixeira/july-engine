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
      2. Resolve model backend (gpu / cpu) from model config.
      3. Route to the right orchestrator.

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
        "video_description": "VISION",
        "video_generation": "VIDEO_GENERATION",
        "entity_extraction": "ENTITY_EXTRACTION",
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
    # Entity Extraction (GLiNER2)
    # ------------------------------------------------------------------

    async def process_entity_extraction(self, payload: dict, headers: dict):
        return await self._dispatch("entity_extraction", payload, headers)

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
            cpu_moe=payload.get("cpu_moe", False),
            n_cpu_moe=payload.get("n_cpu_moe", 0),
            mtp_enabled=payload.get("mtp_enabled", False),
        )

    async def process_video_generation(self, payload: dict, headers: dict):
        return await self._dispatch("video_generation", payload, headers)


bridge = Bridge()
