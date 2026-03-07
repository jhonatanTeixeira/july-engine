import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from ..resource_manager import resource_manager
from ..model_loader import model_loader
from ..routers.calculator import estimate_vram_ram
from ..routers.models import load_models_db

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
                task_types = ["text_chat", "vision_chat", "stt", "tts", "embedding", "pix2pix", "image_generation"]
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

    def _unload_domain_instance(self, instance):
        if hasattr(instance, '_strategy'):
            strategy = instance._strategy
            if hasattr(strategy, 'unload'):
                try:
                    # Some unloads take model_tag (GGUF), others take none (XTTS2)
                    import inspect
                    sig = inspect.signature(strategy.unload)
                    if len(sig.parameters) > 0:
                        strategy.unload(instance.model_tag)
                    else:
                        strategy.unload()
                except Exception as e:
                    logger.warning(f"Error unloading strategy {strategy}: {e}")
            elif hasattr(strategy, 'clear'):
                try:
                    strategy.clear()
                except Exception as e:
                    logger.warning(f"Error clearing strategy {strategy}: {e}")

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
                
                if candidate in self.active_gpu_models:
                    instance = self.active_gpu_models[candidate]
                    self._unload_domain_instance(instance)
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

    def _estimate_required_vram(self, task_type: str, model_tag: str, payload: Any) -> float:
        required_vram_mb = 2000 # default base limit
        
        if task_type in ["text_chat", "vision_chat"]:
            try:
                db = load_models_db()
                if model_tag in db:
                    meta = db[model_tag]
                    params_b = meta.get("num_params")
                    quant = meta.get("quantization")
                    if params_b and quant:
                        headers = payload.get("headers", {})
                        ctx_str = headers.get("x-context-window")
                        
                        effective_n_ctx = meta.get("context_window", 2048)
                        if ctx_str:
                            try:
                                effective_n_ctx = int(ctx_str)
                            except ValueError:
                                pass
                                
                        n_layers = meta.get("num_layers", -1)
                        estimates = estimate_vram_ram(params_b, quant, effective_n_ctx, n_layers)
                        required_vram_mb = estimates["estimated_vram_gb"] * 1024
                        logger.debug(f"GpuOrchestrator: Estimated {required_vram_mb}MB VRAM for {model_tag}")
            except Exception as e:
                logger.error(f"GpuOrchestrator: Failed to estimate VRAM for {model_tag}: {e}")
                
        elif task_type == "tts":
            required_vram_mb = 2500 if "kokoro" not in model_tag.lower() else 1500
        elif task_type in ["pix2pix", "image_generation"]:
            required_vram_mb = 4500
        elif task_type == "stt":
            required_vram_mb = 1500
            
        return required_vram_mb

    def _execute_task_sync(self, task_type: str, payload: Any):
        model_tag = payload.get("model")
        if not model_tag:
            model_tag = os.environ.get("LLM_MODEL")
        
        backend = "gpu"
        
        # Mapping task to resource key
        model_key = "llm"
        if task_type == "vision_chat": model_key = "vision"
        elif task_type == "tts": model_key = "tts"
        elif task_type in ["pix2pix", "image_generation"]: model_key = "pix2pix"

        # Calculate exact VRAM needed and ensure it is available before executing
        required_vram = self._estimate_required_vram(task_type, model_tag, payload)
        self.ensure_resources(model_key, required_vram)

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
                return asyncio.run(mouth.speak(payload))
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
