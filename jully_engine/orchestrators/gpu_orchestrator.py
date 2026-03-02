import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from ..resource_manager import resource_manager
from ..model_loader import model_loader

logger = logging.getLogger("JulyEngine.Orchestrators.GpuOrchestrator")

class GpuOrchestrator:
    """
    Manages GPU-bound tasks using dedicated thread pools and resource management.
    Controls model life-cycle based on VRAM constraints.
    """
    def __init__(self):
        self.parallel_limit = 1
        self.running = False
        self.executors = {}
        self.lock = threading.Lock()
        
        # Resource management
        self.priorities = ["llm", "vision", "tts"]
        self.busy_counts = {k: 0 for k in ["llm", "vision", "tts", "pix2pix"]}
        self.conditions = {k: threading.Condition(self.lock) for k in self.busy_counts.keys()}
        self.active_gpu_models: Dict[str, Any] = {} # model_key -> domain_instance

    async def start(self):
        if not self.running:
            with self.lock:
                self.running = True
                task_types = ["text_chat", "vision_chat", "stt", "tts", "embedding", "pix2pix"]
                for tt in task_types:
                    self.executors[tt] = ThreadPoolExecutor(
                        max_workers=self.parallel_limit, 
                        thread_name_prefix=f"gpu_orchestrator_{tt}"
                    )
                
            logger.info("GpuOrchestrator: Starting up...")
            await self._startup()

    async def stop(self):
        with self.lock:
            self.running = False
            for tt, executor in self.executors.items():
                executor.shutdown(wait=False)
            self.executors = {}
            self.active_gpu_models.clear()

    async def _startup(self):
        """
        Initial model preloading based on STARTUP_MODELS env var.
        """
        raw_startup = os.environ.get("STARTUP_MODELS", "llm,stt,tts,vision").lower()
        startup_models = [m.strip() for m in raw_startup.split(",") if m.strip()]
        
        backend = "gpu"
        # Map human-friendly names to internal model keys and model tags
        for sm in startup_models:
            if sm == "llm":
                model_tag = os.environ.get("LLM_MODEL", "mistral-7b-v0.1.Q4_K_M.gguf")
                logger.info(f"GpuOrchestrator: Preloading Brain ({model_tag})...")
                self.active_gpu_models["llm"] = model_loader.get_brain(backend, model_tag)
            elif sm == "vision":
                model_tag = os.environ.get("VISION_MODEL", "llava-v1.5-7b.Q4_K_M.gguf")
                logger.info(f"GpuOrchestrator: Preloading Eyes ({model_tag})...")
                self.active_gpu_models["vision"] = model_loader.get_eyes(backend, model_tag)
            elif sm == "tts":
                model_tag = "xtts"
                logger.info(f"GpuOrchestrator: Preloading Mouth (xtts)...")
                self.active_gpu_models["tts"] = model_loader.get_mouth(backend, model_tag)

    def mark_busy(self, model_key: str):
        with self.lock:
            if model_key in self.busy_counts:
                self.busy_counts[model_key] += 1

    def mark_idle(self, model_key: str):
        with self.lock:
            if model_key in self.busy_counts:
                self.busy_counts[model_key] = max(0, self.busy_counts[model_key] - 1)
                if self.busy_counts[model_key] == 0:
                    self.conditions[model_key].notify_all()

    def ensure_resources(self, model_key: str, required_vram: float = 4000):
        """
        Unloads models based on priority to free up VRAM.
        """
        with self.lock:
            available = resource_manager.get_available_vram_mb()
            while available < required_vram:
                candidate = None
                # Check transient models first
                if "pix2pix" in self.active_gpu_models and model_key != "pix2pix":
                    candidate = "pix2pix"
                else:
                    # Unload based on reversed priority list
                    for p in reversed(self.priorities):
                        if p in self.active_gpu_models and p != model_key:
                            candidate = p
                            break
                
                if not candidate:
                    break
                
                # If model is busy, wait for it to be idle
                if self.busy_counts.get(candidate, 0) > 0:
                    logger.info(f"GpuOrchestrator: Waiting for {candidate} to become idle...")
                    if not self.conditions[candidate].wait(timeout=30):
                        logger.warning(f"GpuOrchestrator: Timeout waiting for {candidate}, forcing unload.")
                
                logger.info(f"GpuOrchestrator: Unloading {candidate} to free {required_vram}MB")
                # In this refactor, domain classes manage their internal strategy loading
                # We just remove them from active tracking here, strategy will reload if called again
                if candidate in self.active_gpu_models:
                    del self.active_gpu_models[candidate]
                
                resource_manager.clear_memory()
                available = resource_manager.get_available_vram_mb()

    def submit_task(self, task_type: str, payload: Any):
        if not self.running:
            raise RuntimeError("GpuOrchestrator not running")
            
        executor = self.executors.get(task_type)
        if not executor:
            raise ValueError(f"Unknown GPU task type: {task_type}")
        return executor.submit(self._execute_task_sync, task_type, payload)

    def _execute_task_sync(self, task_type: str, payload: Any):
        model_tag = payload.get("model")
        if not model_tag:
            model_tag = os.environ.get("LLM_MODEL")
        
        backend = "gpu"
        
        # Mapping task to resource key
        model_key = "llm"
        if task_type == "vision_chat": model_key = "vision"
        elif task_type == "tts": model_key = "tts"
        elif task_type == "pix2pix": model_key = "pix2pix"

        # Ensure resources (vram)
        # self.ensure_resources(model_key) # logic is complex, keep it simple for now

        self.mark_busy(model_key)
        try:
            logger.info(f"GpuOrchestrator: Executing {task_type} with tag {model_tag}")
            if task_type == "text_chat":
                brain = model_loader.get_brain(backend, model_tag)
                self.active_gpu_models["llm"] = brain
                return asyncio.run(brain.chat(payload))
            elif task_type == "vision_chat":
                eyes = model_loader.get_eyes(backend, model_tag)
                self.active_gpu_models["vision"] = eyes
                return asyncio.run(eyes.analyze(payload))
            elif task_type == "tts":
                mouth = model_loader.get_mouth(backend, model_tag)
                self.active_gpu_models["tts"] = mouth
                temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"temp_gpu_{os.getpid()}_{time.time()}.wav")
                return asyncio.run(mouth.speak(payload, temp_path))
            elif task_type == "stt":
                ears = model_loader.get_ears(backend, model_tag)
                return asyncio.run(ears.listen(payload.get('audio'), payload.get('language'), payload))
            elif task_type == "embedding":
                memory = model_loader.get_memory(backend, model_tag)
                return asyncio.run(memory.embed(payload))
            elif task_type == "pix2pix":
                presence = model_loader.get_presence(backend, model_tag)
                self.active_gpu_models["pix2pix"] = presence
                return asyncio.run(presence.edit(payload))
            elif task_type == "image_generation":
                presence = model_loader.get_presence(backend, model_tag)
                self.active_gpu_models["pix2pix"] = presence
                return asyncio.run(presence.generate(payload))
        finally:
            self.mark_idle(model_key)
            resource_manager.clear_memory()

gpu_orchestrator = GpuOrchestrator()
