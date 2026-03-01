import asyncio
import logging
import os
import time
from typing import Any, Dict
from ..resource_manager import resource_manager
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
                return asyncio.run(brain.chat(payload))
            elif task_type == "vision_chat":
                eyes = model_loader.get_eyes(backend, model_tag)
                return asyncio.run(eyes.analyze(payload))
            elif task_type == "tts":
                mouth = model_loader.get_mouth(backend, model_tag)
                temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"temp_cpu_{os.getpid()}_{time.time()}.wav")
                return asyncio.run(mouth.speak(payload['text'], payload.get('voice'), payload.get('language'), temp_path))
            elif task_type == "stt":
                ears = model_loader.get_ears(backend, model_tag)
                return asyncio.run(ears.listen(payload['audio'], payload.get('language')))
            elif task_type == "embedding":
                memory = model_loader.get_memory(backend, model_tag)
                input_text = payload.get("input")
                return asyncio.run(memory.embed(input_text))
            else:
                raise ValueError(f"Unknown CPU task type: {task_type}")
        except Exception as e:
            logger.error(f"CpuOrchestrator: Task {task_type} failed: {e}")
            raise e

cpu_orchestrator = CpuOrchestrator()
