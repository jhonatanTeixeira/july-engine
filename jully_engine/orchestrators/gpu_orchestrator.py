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

def guess_num_layers(combined_name: str, params: float) -> int:
    """Adivinha o número de layers baseado no tamanho do modelo."""
    if not params:
        return -1 # -1 significa "auto" para o llama.cpp
        
    combined_name = combined_name.lower()
    
    # Família 7B - 8B
    if 7 <= params <= 9:
        if "gemma" in combined_name and params >= 9:
            return 42 # Gemma 2 9B
        return 32
        
    # Família 0.5B - 3B
    if params < 3:
        if "qwen" in combined_name and params < 1:
            return 24 # Qwen 0.5B
        if "qwen" in combined_name and 1 <= params <= 2:
            return 28 # Qwen 1.5B
        if "gemma" in combined_name:
            return 18 # Gemma 2B
        if "phi" in combined_name:
            return 32 # Phi-2 / Phi-3 Mini
        return 24
        
    # Família 13B - 14B
    if 12 <= params <= 15:
        if "qwen" in combined_name:
            return 48
        return 40
        
    # Família 32B - 35B
    if 30 <= params <= 35:
        return 64
        
    # Família 70B+
    if params >= 70:
        return 80

    return -1

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
        self.priorities = ["pix2pix", 'tts', 'stt', 'vision', 'llm']
        self.busy_counts = {k: 0 for k in ["llm", "vision", "tts", "pix2pix"]}
        self.conditions = {k: threading.Condition(self.lock) for k in self.busy_counts.keys()}
        self.active_gpu_models: Dict[str, Any] = {} 

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
            # await self._startup()

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
                logger.info(f"GpuOrchestrator: Preloading Mouth (xtts)...")
                self.active_gpu_models["tts"] = model_loader.get_mouth(backend, 'xtts')

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

    def ensure_resources(self, model_key: str, required_vram: float = 4000) -> float:
        """
        Unloads models based on priority to free up VRAM.
        Returns the final available VRAM.
        """
        with self.lock:
            available = resource_manager.get_available_vram_mb()
            
            if available >= required_vram:
                return available
            
            busy_candidates = []
            
            for candidate in self.priorities:
                if candidate not in self.active_gpu_models:
                    continue
                
                if self.busy_counts.get(candidate, 0) > 0:
                    busy_candidates.append(candidate)
                    continue
                
                self._unload_domain_instance(self.active_gpu_models[candidate])
                
                available = resource_manager.get_available_vram_mb()
                
                if available >= required_vram:
                    return available

            for busy in busy_candidates:
                if not self.conditions[busy].wait(timeout=300):
                    logger.warning(f"GpuOrchestrator: Timeout waiting for {candidate}, forcing unload.")
                    continue
                
                self._unload_domain_instance(self.active_gpu_models[busy])
                
                if (available := resource_manager.get_available_vram_mb()) >= required_vram:
                    return available
                
            return available

    def submit_task(self, task_type: str, payload: Any):
        if not self.running: raise RuntimeError("GpuOrchestrator not running")
        executor = self.executors.get(task_type)
        if not executor: raise ValueError(f"Unknown GPU task type: {task_type}")
        return executor.submit(self._execute_task_sync, task_type, payload)

    def _execute_task_sync(self, task_type: str, payload: Any):
        from fastapi import HTTPException
        
        model_tag = payload.get("model") or os.environ.get("LLM_MODEL")
        backend = "gpu"
        
        model_key = "llm"
        
        if task_type == "vision_chat":
            model_key = "vision"
        elif task_type == "tts":
            model_key = "tts"
        elif task_type in ["pix2pix", "image_generation"]:
            model_key = "pix2pix"

        # 1. Resource pre-check and unloading
        db = load_models_db()
        meta = db.get(model_tag, {})
        params_b = meta.get("num_params", 0)
        quant = meta.get("quantization", "Q4_K_M")
        
        headers = payload.get("headers", {})
        effective_n_ctx = int(headers.get("x-context-window") or meta.get("context_window") or 2048)
        
        n_layers = meta.get("num_layers", -1)
        
        if meta.get('model_type', 'text') in ['text', 'vision']:
            if n_layers == -1:
                n_layers = guess_num_layers(model_tag + meta.get("filename", ""), params_b)
            
            estimates = estimate_vram_ram(params_b, quant, effective_n_ctx, n_layers)
            required_vram_mb = estimates["estimated_vram_gb"] * 1024
        else:
            required_vram_mb = meta.get('estimated_vram', 0)
            
        if (existing_instance := self.active_gpu_models.get(model_key, None)) and getattr(existing_instance, 'model_tag', None) == model_tag:
            required_vram_mb = 0

        # 2. Try to free up memory
        available_vram = self.ensure_resources(model_key, required_vram_mb)

        # 3. Iterative layer optimization if it still doesn't fit
        if available_vram < required_vram_mb and n_layers > 0:
            logger.info(f"GpuOrchestrator: Model {model_tag} ({required_vram_mb:.2f}MB) too big for VRAM ({available_vram:.2f}MB). Decrementing layers...")
            
            while meta.get('model_type', 'text') in ['text', 'vision'] and n_layers > 0 and available_vram < required_vram_mb:
                n_layers -= 1
                estimates = estimate_vram_ram(params_b, quant, effective_n_ctx, n_layers)
                required_vram_mb = estimates["estimated_vram_gb"] * 1024
            
            logger.info(f"GpuOrchestrator: Optimized model to {n_layers} layers ({required_vram_mb:.2f}MB required)")
            
            # Safety margin: decrement one more if possible
            if n_layers > 0:
                n_layers -= 1
                
            payload["num_layers"] = n_layers

        if available_vram < required_vram_mb:
            raise HTTPException(status_code=422, detail=f"Insufficient VRAM even with 0 layers. Required: {required_vram_mb:.2f}MB, Available: {available_vram:.2f}MB.")

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
