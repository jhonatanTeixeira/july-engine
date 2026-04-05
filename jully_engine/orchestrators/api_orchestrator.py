import asyncio
import logging
from typing import Any, Dict
from ..model_loader import model_loader

logger = logging.getLogger("JulyEngine.Orchestrators.ApiOrchestrator")

class ApiOrchestrator:
    """
    Manages API-based tasks via litellm domain classes.
    """
    def __init__(self):
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    async def submit_task(self, task_type: str, payload: Any):
        model_tag = payload.get("model")
        backend = "api"

        try:
            if task_type == "text_chat":
                brain = model_loader.get_brain(backend, model_tag)
                return await brain.chat(payload)
            elif task_type == "vision_chat":
                eyes = model_loader.get_eyes(backend, model_tag)
                return await eyes.analyze(payload)
            elif task_type == "tts":
                mouth = model_loader.get_mouth(backend, model_tag)
                return await mouth.speak(payload)
            elif task_type == "stt":
                ears = model_loader.get_ears(backend, model_tag)
                return await ears.listen(payload.get('audio'), payload.get('language'), payload)
            elif task_type == "embedding":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.embed(payload)
            elif task_type == "rag_add":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.add_to_rag(payload.get("text"), payload.get("metadata"), payload.get("collection", "july_memory"))
            elif task_type == "rag_batch_add":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.add_batch_to_rag(payload.get("documents", []), payload.get("collection", "july_memory"))
            elif task_type == "rag_search":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.search(payload.get("query"), payload.get("top_k", 3), payload.get("collection", "july_memory"))
            elif task_type == "rag_vector_add":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.add_vector_to_rag(payload.get("vector"), payload.get("text", ""), payload.get("metadata"), payload.get("collection", "july_memory"))
            elif task_type == "rag_search_details":
                memory = model_loader.get_memory(backend, model_tag)
                vector = payload.get("vector")
                if vector:
                    return await memory.search_with_details_vector(vector, payload.get("top_k", 3), payload.get("collection", "july_memory"))
                else:
                    emb = await memory.embed({"input": payload.get("query")})
                    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list): emb = emb[0]
                    return await memory.search_with_details_vector(emb, payload.get("top_k", 3), payload.get("collection", "july_memory"))
            elif task_type == "rag_update":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.update_embedding(str(payload.get("id")), payload.get("vector"))
            elif task_type == "pix2pix" or task_type == 'image_edit':
                presence = model_loader.get_presence(backend, model_tag)
                return await presence.edit(payload)
            elif task_type == "image_generation":
                presence = model_loader.get_presence(backend, model_tag)
                return await presence.generate(payload)
            elif task_type == "image_resize":
                presence = model_loader.get_presence(backend, model_tag)
                return await presence.resize(payload)
            elif task_type == "search_web":
                world = model_loader.get_world(backend, model_tag)
                return await world.search_web(payload)
            elif task_type == "search_code":
                world = model_loader.get_world(backend, model_tag)
                return await world.search_code(payload)
            else:
                raise ValueError(f"Unknown API task type: {task_type}")
        except Exception as e:
            logger.error(f"ApiOrchestrator: Task {task_type} failed: {e}")
            raise e

api_orchestrator = ApiOrchestrator()
