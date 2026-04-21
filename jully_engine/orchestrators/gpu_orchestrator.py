import asyncio
import threading
import time
import inspect
from queue import Queue, Empty
from typing import Any, AsyncGenerator, Union, Dict
import logging

logger = logging.getLogger('JulyEngne.Orchestrators.GpuOrchestrator')


class GpuContext:
    def __init__(self):
        from ..resource_manager import resource_manager
        self.state = {}
        self.state_lock = threading.Lock()
        self.orchestrator_lock = asyncio.Lock() 
        self.condition = asyncio.Condition() 
        self.resource_manager = resource_manager

    def get_runner(self, task_type: str) -> 'Runner':
        return self.state.get(task_type, {}).get("runner")

    def release_task(self, task_type: str):
        with self.state_lock:
            if task_type in self.state:
                self.state[task_type]['status'] = 'idle'
                self.state[task_type]['usage_count'] = 0
                self.state[task_type]['runner'] = None

    def mark_busy(self, task_type: str, runner: 'Runner'):
        with self.state_lock:
            if task_type not in self.state:
                self.state[task_type] = {'status': 'idle', 'usage_count': 0, 'runner': runner}
            
            self.state[task_type]['status'] = 'busy'
            self.state[task_type]['usage_count'] += 1
            self.state[task_type]['runner'] = runner

    async def mark_idle(self, task_type: str):
        async with self.condition:
            with self.state_lock:
                if task_type in self.state:
                    self.state[task_type]['usage_count'] -= 1
                    if self.state[task_type]['usage_count'] <= 0:
                        self.state[task_type]['usage_count'] = 0
                        self.state[task_type]['status'] = 'idle'
            self.condition.notify_all()

    def is_truly_idle(self, task_type: str):
        with self.state_lock:
            data = self.state.get(task_type)
            return data and data['status'] == 'idle' and data['usage_count'] == 0

    def is_loaded(self, task_type: str):
        with self.state_lock:
            return task_type in self.state

    def get_free_vram(self):
        return self.resource_manager.get_available_vram_mb()

    def garbage_collection(self):
        self.resource_manager.clear_memory()

class Runner:
    def __init__(self, task_type: str, model: dict, context: GpuContext):
        from ..model_loader import model_loader # Import interno para evitar circularidade
        self.task_type = task_type
        self.model = model
        self.context = context
        self.model_loader = model_loader
        self.is_running = True
        self.state_lock = threading.Lock()
        self.unload_priority = [
            'stt', 'tts', 'embeddings', 'rag_add', 'rag_batch_add', 
            'rag_search', 'rag_update', 'pix2pix', 'image_generation', 'image_resize', 'vision_chat', 'text_chat'
        ]

    def get_domain(self):
        if self.task_type == "text_chat": 
            return self.model_loader.get_brain('gpu', self.model)
        if self.task_type == "vision_chat": 
            return self.model_loader.get_eyes('gpu', self.model)
        if self.task_type == "tts": 
            return self.model_loader.get_mouth('gpu', self.model)
        if self.task_type == "stt": 
            return self.model_loader.get_ears('gpu', self.model)
        if self.task_type in ["pix2pix", "image_generation", "image_resize"]:
            return self.model_loader.get_presence('gpu', self.model)

        return self.model_loader.get_memory('gpu', self.model)

    async def run_task(self, payload):
        task_type = self.task_type
        domain = self.get_domain()

        if task_type == "text_chat": 
            result = domain.chat(payload)
        elif task_type == "vision_chat": 
            result = domain.analyze(payload)
        elif task_type == "tts": 
            result = domain.speak(payload)
        elif task_type == "stt": 
            result = domain.listen(payload.get('audio'), payload.get('language'), payload)
        elif task_type in ["embedding", "embeddings"]: 
            result = domain.embed(payload)
        elif task_type == "rag_add": 
            result = domain.add_to_rag(payload.get("text"), payload.get("metadata"), payload.get("collection", "july_memory"))
        elif task_type == "rag_batch_add": 
            result = domain.add_batch_to_rag(payload.get("documents", []), payload.get("collection", "july_memory"))
        elif task_type == "rag_search": 
            vector = payload.get("vector")
            if vector:
                result = domain.search_with_details_vector(vector, payload.get("top_k", 3), payload.get("collection", "july_memory"))
            else:
                result = domain.search(payload.get("query"), payload.get("top_k", 3), payload.get("collection", "july_memory"))
        elif task_type == "rag_vector_add": 
            result = domain.add_vector_to_rag(payload.get("vector"), payload.get("text", ""), payload.get("metadata"), payload.get("collection", "july_memory"))
        elif task_type == "rag_update": 
            result = domain.update_embedding(str(payload.get("id")), payload.get("vector"), payload.get("collection", "july_memory"))
        elif task_type == "rag_delete": 
            result = domain.delete_from_rag(payload.get("ids", []), payload.get("collection", "july_memory"))
        elif task_type == "rag_list": 
            result = domain.list_rag_metadata(payload.get("collection", "july_memory"))
        elif task_type == "rag_smart_search": 
            result = domain.smart_search(payload)
        elif task_type == "image_generation": 
            result = domain.generate(payload)
        elif task_type == "pix2pix": 
            result = domain.edit(payload)
        elif task_type == "image_resize": 
            result = domain.resize(payload)

        if inspect.iscoroutine(result):
            result = await result

        return result

    def unload_next(self, required_vram: int):
        """Busca o próximo modelo idle na prioridade e descarrega."""
        for task in self.unload_priority:
            if self.context.is_loaded(task) and self.context.is_truly_idle(task):
                logger.info(f"📦 [Orchestrator] Descarregando {task} para liberar espaço...")
                runner = self.context.get_runner(task)
                runner.unload()
                
                if self.context.get_free_vram() >= required_vram:
                    return True

        return False

    def unload(self, timeout=60):
        with self.state_lock:
            time = 0

            while not self.context.is_truly_idle(self.task_type) and time <= timeout:
                time += 1
                time.sleep(1)

            if time >= timeout:
                raise Exception(f"Timeout na GPU: {self.task_type} não está idle após {timeout}s.")

            self.get_domain().unload()
            self.context.garbage_collection()
            self.context.release_task(self.task_type)

    async def wait_for_free_vram(self, required_vram: int, timeout: int = 60):
        start_time = time.time()

        async with self.context.condition:
            while self.context.get_free_vram() < required_vram:
                # Se algo ficou idle enquanto esperávamos, tentamos dar unload
                if any(self.context.is_truly_idle(t) for t in self.context.state):
                    return 

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"VRAM insuficiente após {timeout}s.")

                try:
                    await asyncio.wait_for(self.context.condition.wait(), timeout=(timeout - elapsed))
                except asyncio.TimeoutError:
                    raise TimeoutError("Timeout na fila da GPU.")

    async def run(self, payload: dict):
        async with self.context.orchestrator_lock:
            self.context.garbage_collection()

            domain = self.get_domain()
            runner = self.context.get_runner(self.task_type)

            if self.context.is_loaded(self.task_type) and runner.model != self.model:
                runner.unload()

            elif not domain.is_loaded():
                required = await domain.get_required_vram(payload)

                while self.context.get_free_vram() < required:
                    # Tenta descarregar o que já está parado
                    unloaded = await self.unload_next(required)
                    
                    if not unloaded:
                        if hasattr(domain._strategy, "decrement_layers"):
                            domain._strategy.decrement_layers()
                            
                            while await domain.get_required_vram(payload) > self.context.get_free_vram():
                                domain._strategy.decrement_layers()
                        else:
                            # Se nada pode sair, espera alguém terminar de usar (mark_idle)
                            await self.wait_for_free_vram(required)
                
                domain.load()

            self.context.mark_busy(self.task_type, self)
        
        try:
            return await self.run_task(payload)
        finally:
            # Libera o contador mas MANTÉM o modelo carregado ("quente")
            await self.context.mark_idle(self.task_type)
            self.context.garbage_collection()


class GpuOrchestrator:
    def __init__(self):
        # Passamos o resource_manager para o contexto gerenciar VRAM
        self.context = GpuContext()
        self.slots: Dict[str, Runner] = {}

    async def start(self):
        """Inicialização de serviços globais da GPU se necessário."""
        logger.info("🎸 [GpuOrchestrator] Sistema de GPU online.")

    async def stop(self):
        """Cleanup de todos os modelos carregados."""
        logger.info(" cleanup de memória da GPU...")
        self.context.garbage_collection()

    async def submit_task(self, task_type: str, payload: Dict) -> Union[Any, AsyncGenerator[Any, None]]:
        """
        Interface principal chamada pelo InferenceHelper.
        Cria um Runner temporário para orquestrar a memória e executar a tarefa.
        """
        from ..services.models_service import model_service

        model = payload.get("model") or model_service.resolve_by_task_type(task_type)
        slot = f"{task_type}_{model}"

        if slot not in self.slots:
            self.slots[slot] = Runner(task_type, model, self.context)

        try:
            result = await self.slots[slot].run(payload)
            return result
        except Exception as e:
            logger.error(f"❌ [GpuOrchestrator] Erro ao processar {task_type}: {str(e)}")
            raise e


gpu_orchestrator = GpuOrchestrator()