import asyncio
import logging
import os
import time
from typing import Any, Dict
from ..model_loader import model_loader

logger = logging.getLogger("JulyEngine.Orchestrators.CpuOrchestrator")

class CpuOrchestrator:
    """
    Manages CPU-bound tasks with preloading and resource-based throttling.
    Everything here is CPU bound services.
    """
    def __init__(self):
        self.running = False

    async def start(self):
        if not self.running:
            self.running = True
            logger.info("CpuOrchestrator: Starting up and preloading CPU models...")
            await self._preload_models()

    async def stop(self):
        self.running = False

    async def _preload_models(self):
        """
        Preloads models specifically designed for CPU (like FastVLM, Emotion, BgeMicro).
        """
        raw_startup = os.environ.get("STARTUP_MODELS", "llm,stt,tts,vision").lower()
        startup_list = [m.strip() for m in raw_startup.split(",") if m.strip()]
        
        backend = "cpu"
        # Preload specific CPU-optimized models if requested
        if "vision" in startup_list:
            logger.info("CpuOrchestrator: Preloading FastVLM and Emotion...")
            model_loader.get_eyes(backend, "fastvlm")
            model_loader.get_eyes(backend, "emotion")
        
        if "stt" in startup_list:
            logger.info("CpuOrchestrator: Preloading FasterWhisper (CPU)...")
            model_loader.get_ears(backend, "faster-whisper")

    def _throttle(self):
        """
        Throttles execution if CPU or RAM usage is too high, making requests wait.
        """
        from ..resource_manager import resource_manager
        
        while resource_manager.get_cpu_usage() > 90 or resource_manager.get_ram_usage() > 95:
            logger.warning("CpuOrchestrator: System overloaded (CPU/RAM > 90/95%), throttling request...")
            time.sleep(1)

    def submit_task(self, task_type: str, payload: Any):
        """
        Submits a task to be run on CPU. Mimics Future interface via run_in_executor.
        """
        if not self.running:
            raise RuntimeError("CpuOrchestrator not running")
            
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, self._execute_task_sync, task_type, payload)

    def _execute_task_sync(self, task_type: str, payload: Any):
        self._throttle()
        
        model_tag = payload.get("model")
        backend = "cpu"

        try:
            logger.info(f"CpuOrchestrator: Executing {task_type} with tag {model_tag}")
            if task_type == "text_chat":
                brain = model_loader.get_brain(backend, model_tag)
                return brain.chat(payload)
            elif task_type == "vision_chat":
                eyes = model_loader.get_eyes(backend, model_tag)
                return eyes.analyze(payload)
            elif task_type == "tts":
                mouth = model_loader.get_mouth(backend, model_tag)
                return mouth.speak(payload)
            elif task_type == "stt":
                ears = model_loader.get_ears(backend, model_tag)
                return ears.listen(payload.get('audio'), payload.get('language'), payload)
            elif task_type == "embedding":
                memory = model_loader.get_memory(backend, model_tag)
                return memory.embed(payload)
            elif task_type == "rag_add":
                memory = model_loader.get_memory(backend, model_tag)
                return asyncio.run(memory.add_to_rag(payload.get("text"), payload.get("metadata"), payload.get("collection", "july_memory"), payload.get("id")))
            elif task_type == "rag_batch_add":
                memory = model_loader.get_memory(backend, model_tag)
                return asyncio.run(memory.add_batch_to_rag(payload.get("documents", []), payload.get("collection", "july_memory")))
            elif task_type == "rag_search":
                memory = model_loader.get_memory(backend, model_tag)
                vector = payload.get("vector")
                if vector:
                    return asyncio.run(memory.search_with_details_vector(vector, payload.get("top_k", 3), payload.get("collection", "july_memory")))
                else:
                    return asyncio.run(memory.search(payload.get("query"), payload.get("top_k", 3), payload.get("collection", "july_memory")))
            elif task_type == "rag_vector_add":
                memory = model_loader.get_memory(backend, model_tag)
                return asyncio.run(memory.add_vector_to_rag(payload.get("vector"), payload.get("text", ""), payload.get("metadata"), payload.get("collection", "july_memory")))
            elif task_type == "rag_update":
                memory = model_loader.get_memory(backend, model_tag)
                return asyncio.run(memory.update_embedding(str(payload.get("id")), payload.get("vector"), payload.get("collection", "july_memory")))
            elif task_type in ["pix2pix", "image_generation"]:
                presence = model_loader.get_presence(backend, model_tag)
                return asyncio.run(presence.generate(payload))
            elif task_type == "image_resize":
                presence = model_loader.get_presence(backend, model_tag)
                return asyncio.run(presence.resize(payload))
            else:
                raise ValueError(f"Unknown CPU task type: {task_type}")
        except Exception as e:
            logger.error(f"CpuOrchestrator: Task {task_type} failed: {e}")
            raise e

cpu_orchestrator = CpuOrchestrator()
