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
                self.state[slot_name] = {'status': 'idle', 'usage_count': 0, 'runner': runner}
            
            self.state[slot_name]['status'] = 'busy'
            self.state[slot_name]['usage_count'] += 1
            self.state[slot_name]['runner'] = runner

    async def mark_idle(self, slot_name: str):
        async with self.condition:
            with self.state_lock:
                if slot_name in self.state:
                    self.state[slot_name]['usage_count'] -= 1
                    if self.state[slot_name]['usage_count'] <= 0:
                        self.state[slot_name]['usage_count'] = 0
                        self.state[slot_name]['status'] = 'idle'
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

    async def get_free_ram(self):
        raise NotImplementedError()


class GpuContext(BaseContext):
    def __new__(cls, *args, **kwargs):
        return super(GpuContext, cls).__new__(cls)

    def get_free_ram(self):
        return self.resource_manager.get_available_vram_mb()


class CpuContext(BaseContext):
    def __new__(cls, *args, **kwargs):
        return super(CpuContext, cls).__new__(cls)

    def get_free_ram(self):
        return self.resource_manager.get_available_ram_mb()


class Runner:
    def __init__(self, task_type: str, model: dict, context: BaseContext):
        from .model_loader import model_loader

        self.task_type = task_type
        self.model = model
        self.model_tag = model.get("alias") or model.get("model")
        self.slot_name = f'{task_type}_{self.model_tag}'
        self.context = context
        self.model_loader = model_loader
        self.is_running = False
        self.state_lock = threading.Lock()

    def get_model(self) -> BaseModel:
        return self.model_loader.get(
            self.task_type,
            'cpu' if isinstance(self.context, CpuContext) else 'gpu',
            self.model
        )

    async def wait_for_resources(self, required: float, timeout: float = 30):
        start_time = time.time()
        while await self.context.get_free_ram() < required:
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
        model = self.get_model()
        model.unload()

    async def run(self, payload: dict):
        model_alias = self.model_tag
            
        model_lock = self.context.get_model_lock(model_alias)

        async with model_lock:
            async with self.context.orchestrator_lock:
                self.context.garbage_collection()

                domain = self.get_model()
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
                                    raise MemoryError(f"VRAM insuficiente para {self.task_type}. Requerido: {required}MB, Livre: {self.context.get_free_vram()}MB")
                            
                            # Atualiza o valor 'required' após o decremento
                            required = await domain.get_required_vram(payload)
                        else:
                            # Se o modelo não suporta decremento (ex: Flux), apenas espera por liberação externa
                            await self.wait_for_resources(required, timeout=10)
                            if self.context.get_free_ram() < required:
                                raise MemoryError(f"VRAM insuficiente para {self.task_type}. Requerido: {required}MB, Livre: {self.context.get_free_vram()}MB")

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
    def __init__(self, context: BaseContext):
        self.context = context

    async def start(self):
        pass

    async def stop(self):
        pass

    async def submit_task(self, task_type: str, payload: dict):
        from .services.models_service import model_service
        
        model_alias = payload.get("model")
        model_meta = model_service.resolve_by_settings(model_alias)
        
        if not model_meta:
            raise ValueError(f"Orchestrator: Model '{model_alias}' not found in persistence.")

        runner = Runner(task_type, model_meta, self.context)
        return await runner.run(payload)

    async def unload_model(self, model_alias: str):
        slots_to_remove = []
        with self.context.state_lock:
            for slot_name, data in self.context.state.items():
                runner_meta = data["runner"].model
                if runner_meta.get("model_alias") == model_alias or runner_meta.get("alias") == model_alias:
                    slots_to_remove.append(slot_name)
        
        for slot_name in slots_to_remove:
            with self.context.state_lock:
                data = self.context.state.get(slot_name)
                if data:
                    data["runner"].unload()
                    del self.context.state[slot_name]
                
                from .model_loader import model_loader
                model_loader.delete_instance(self.context.context_type, model_alias)


gpu_orchestrator = Orchestrator(GpuContext())
cpu_orchestrator = Orchestrator(CpuContext())
