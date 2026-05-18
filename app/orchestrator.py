import logging
import asyncio
import threading
import time
import uuid
import os
import sys
import builtins
from typing import Dict, Any, Optional, AsyncGenerator
from .context import request_id_var
from .models.base_model import BaseModel


logger = logging.getLogger("JulyEngine.Orchestrator")


class ReentrantModelLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0

    async def acquire(self):
        rid = request_id_var.get()
        if rid and self._owner == rid:
            self._count += 1
            return
        await self._lock.acquire()
        self._owner = rid
        self._count = 1

    def release(self):
        rid = request_id_var.get()
        if rid and self._owner == rid:
            self._count -= 1
            if self._count == 0:
                self._owner = None
                self._lock.release()
        else:
            # Fallback para chamadas sem request_id
            self._owner = None
            self._count = 0
            if self._lock.locked():
                self._lock.release()

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        self.release()


class BaseContext:
    def __init__(self):
        from .resource_manager import resource_manager
        self.state = {}
        self.state_lock = threading.Lock()
        self.orchestrator_lock = asyncio.Lock()
        self.condition = asyncio.Condition()
        self.resource_manager = resource_manager
        self.model_locks = {}
        self.model_locks_lock = threading.Lock()

    @property
    def context_type(self) -> str:
        return "base"

    def get_model_lock(self, model_alias: str):
        with self.model_locks_lock:
            if model_alias not in self.model_locks:
                self.model_locks[model_alias] = ReentrantModelLock()
            return self.model_locks[model_alias]

    def get_runner(self, slot_name: str) -> 'Runner':
        return self.state.get(slot_name, {}).get("runner")

    def release_task(self, slot_name: str):
        with self.state_lock:
            if slot_name in self.state:
                self.state[slot_name]['status'] = 'idle'
                self.state[slot_name]['usage_count'] = 0
                self.state[slot_name]['runner'] = None

    def mark_busy(self, slot_name: str, runner: 'Runner'):
        with self.state_lock:
            if slot_name not in self.state:
                self.state[slot_name] = {'status': 'idle', 'usage_count': 0, 'runner': runner, 'last_used': time.time()}

            self.state[slot_name]['status'] = 'busy'
            self.state[slot_name]['usage_count'] += 1
            self.state[slot_name]['runner'] = runner
            self.state[slot_name]['last_used'] = time.time()

    async def mark_idle(self, slot_name: str):
        async with self.condition:
            with self.state_lock:
                if slot_name in self.state:
                    self.state[slot_name]['usage_count'] -= 1
                    if self.state[slot_name]['usage_count'] <= 0:
                        self.state[slot_name]['usage_count'] = 0
                        self.state[slot_name]['status'] = 'idle'
                        self.state[slot_name]['last_used'] = time.time()
            self.condition.notify_all()

    def is_truly_idle(self, slot_name: str):
        with self.state_lock:
            data = self.state.get(slot_name)
            return data and data['status'] == 'idle' and data['usage_count'] == 0

    def is_loaded(self, slot_name: str):
        with self.state_lock:
            data = self.state.get(slot_name)
            return data is not None and data.get("runner") is not None

    def garbage_collection(self):
        self.resource_manager.clear_memory()

    def get_free_ram(self):
        raise NotImplementedError()


class GpuContext(BaseContext):
    def __new__(cls, *args, **kwargs):
        return super(GpuContext, cls).__new__(cls)

    @property
    def context_type(self) -> str:
        return "gpu"

    def get_free_ram(self):
        return self.resource_manager.get_available_vram_mb()


class CpuContext(BaseContext):
    def __new__(cls, *args, **kwargs):
        return super(CpuContext, cls).__new__(cls)

    @property
    def context_type(self) -> str:
        return "cpu"

    def get_free_ram(self):
        return self.resource_manager.get_available_ram_mb()


class ApiContext(BaseContext):
    @property
    def context_type(self) -> str:
        return "api"

    def get_free_ram(self):
        # API context doesn't track local RAM/VRAM
        return float('inf')


class Runner:
    def __init__(self, task_type: str, model_tag: str | None, context: BaseContext | None = None):
        from .model_loader import model_loader

        backend = None if context is None else context.context_type

        self.task_type = task_type
        self.model = model_loader.get(task_type, backend, model_tag)
        self.model_tag = model_tag or self.model.model_id
        self.slot_name = f'{task_type}_{self.model_tag}'

        
        self.context: BaseContext = context

        if not self.context:
            backend = self.model.backend
            _ctx_map = {"gpu": GpuContext, "cpu": CpuContext, "api": ApiContext}
            ctx_cls = _ctx_map.get(backend, ApiContext)
            self.context = ctx_cls()
        
        self.model_loader = model_loader
        self.is_running = False
        self.state_lock = threading.Lock()

    async def wait_for_resources(self, required: float, timeout: float = 30):
        start_time = time.time()

        while self.context.get_free_ram() < required:
            if time.time() - start_time > timeout:
                return False
            async with self.context.condition:
                try:
                    await asyncio.wait_for(self.context.condition.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
        
        return True

    async def unload_next(self, required: float):
        """Tenta descarregar o modelo menos usado (LRU) que esteja IDLE."""
        candidates = []
        with self.context.state_lock:
            for name, data in self.context.state.items():
                if data["status"] == "idle" and name != self.slot_name:
                    candidates.append((data["last_used"], name, data["runner"]))
        
        if not candidates:
            return False
            
        candidates.sort() # Ordena por last_used (mais antigo primeiro)
        
        _, name, runner = candidates[0]
        logger.info(f"Orchestrator[{self.context.context_type}]: Unloading {name} to free resources.")
        runner.unload()
        
        with self.context.state_lock:
            self.context.state.pop(name, None)
            
        self.context.garbage_collection()
        return True

    def unload(self):
        model = self.model
        model.unload()

    async def run(self, payload: dict):
        model_alias = self.model_tag
            
        model_lock = self.context.get_model_lock(model_alias)

        async with model_lock:
            async with self.context.orchestrator_lock:
                self.context.garbage_collection()

                domain = self.model
                required = 0

                if not self.context.is_loaded(self.slot_name):
                    required = await domain.get_required_vram(payload)
                
                while self.context.get_free_ram() < required:
                    # Tenta descarregar o que já está parado
                    unloaded = await self.unload_next(required)
                    
                    if not unloaded:
                        # Se não há mais nada para descarregar, tentamos reduzir camadas do próprio modelo (se GGUF)
                        if hasattr(domain, "decrement_layers"):
                            success = domain.decrement_layers()
                            if not success:
                                # Se chegou em 0 camadas e ainda não cabe, esperamos
                                await self.wait_for_resources(required, timeout=10)
                                if self.context.get_free_ram() < required:
                                    raise MemoryError(f"VRAM insuficiente para {self.task_type}. Requerido: {required}MB, Livre: {self.context.get_free_ram()}MB")
                            
                            # Atualiza o valor 'required' após o decremento
                            required = await domain.get_required_vram(payload)
                        else:
                            # Se o modelo não suporta decremento (ex: Flux), apenas espera por liberação externa
                            await self.wait_for_resources(required, timeout=10)
                            if self.context.get_free_ram() < required:
                                raise MemoryError(f"VRAM insuficiente para {self.task_type}. Requerido: {required}MB, Livre: {self.context.get_free_ram()}MB")

                if not domain.is_loaded():
                    domain.load()

                self.context.mark_busy(self.slot_name, self)
        
        result = None
        is_stream = False

        try:
            result = await domain.run(payload)

            if hasattr(result, '__aiter__'):
                is_stream = True
                
                async def generator_wrapper(gen, context, slot_name):
                    try:
                        async for chunk in gen:
                            yield chunk
                    finally:
                        await context.mark_idle(slot_name)
                        context.garbage_collection()
                
                return generator_wrapper(result, self.context, self.slot_name)
            
            return result
        finally:
            if not is_stream:
                # Libera o contador mas MANTÉM o modelo carregado ("quente")
                await self.context.mark_idle(self.slot_name)
                self.context.garbage_collection()


class Orchestrator:
    def __init__(self):
        self.contexts: Dict[str, BaseContext] = {}
        # Singleton contexts per backend — shared across all models on the same hardware.
        # This is the critical fix: without singletons, every request creates a fresh
        # context with empty state, so is_loaded() always returns False and the model
        # tries to reload itself while still occupying VRAM.
        self._backend_contexts: Dict[str, BaseContext] = {
            'gpu': GpuContext(),
            'cpu': CpuContext(),
            'api': ApiContext(),
        }

    async def start(self):
        pass

    async def stop(self):
        pass

    async def submit_task(self, task_type: str, payload: dict):
        model_alias: str | None = payload.get("model")

        # Pre-select context only when x-backend is explicit; Runner.__init__ will
        # determine the backend from model metadata otherwise.
        backend = payload.get("headers", {}).get("x-backend")
        context = self._backend_contexts.get(backend) if backend else None

        runner = Runner(task_type, model_alias, context)

        # Always replace runner.context with the shared singleton for that backend type.
        # This guarantees slot state (is_loaded, mark_busy, unload_next) is consistent
        # across all requests for all models on the same backend.
        runner.context = self._backend_contexts[runner.context.context_type]
        self.contexts[runner.model_tag] = runner.context

        return await runner.run(payload)

    async def unload_model(self, model_alias: str):
        slots_to_remove = []

        if not model_alias in self.contexts:
            return

        with self.contexts[model_alias].state_lock:
            for slot_name, data in self.contexts[model_alias].state.items():
                runner_meta = data["runner"].model.meta
                if runner_meta.get("model_alias") == model_alias or runner_meta.get("alias") == model_alias:
                    slots_to_remove.append(slot_name)
        
        for slot_name in slots_to_remove:
            with self.contexts[model_alias].state_lock:
                data = self.contexts[model_alias].state.get(slot_name)
                if data:
                    data["runner"].unload()
                    del self.contexts[model_alias].state[slot_name]
                
                from .model_loader import model_loader
                model_loader.delete_instance(self.contexts[model_alias].context_type, model_alias)


orchestrator = Orchestrator()