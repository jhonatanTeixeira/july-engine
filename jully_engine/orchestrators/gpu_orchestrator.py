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
            data = self.state.get(task_type)
            return data and data['status'] == 'busy'

    def get_free_vram(self):
        return self.resource_manager.get_available_vram_mb()

    def garbage_collection(self):
        self.resource_manager.clear_memory()

class Runner:
    def __init__(self, task_type: str, payload: dict, context: GpuContext):
        from ..model_loader import model_loader # Import interno para evitar circularidade
        self.task_type = task_type
        self.payload = payload
        self.context = context
        self.model_loader = model_loader
        self.is_running = True
        self.unload_priority = [
            'stt', 'tts', 'embeddings', 'rag_add', 'rag_batch_add', 
            'rag_search', 'rag_update', 'pix2pix', 'image_generation', 'image_resize', 'vision_chat', 'text_chat'
        ]

    def get_domain(self, backend, model_tag):
        if self.task_type == "text_chat": 
            return self.model_loader.get_brain(backend, model_tag)
        if self.task_type == "vision_chat": 
            return self.model_loader.get_eyes(backend, model_tag)
        if self.task_type == "tts": 
            return self.model_loader.get_mouth(backend, model_tag)
        if self.task_type == "stt": 
            return self.model_loader.get_ears(backend, model_tag)
        if self.task_type in ["pix2pix", "image_generation", "image_resize"]:
            return self.model_loader.get_presence(backend, model_tag)
        if self.task_type in ["search_web", "search_code"]:
            return self.model_loader.get_world(backend, model_tag)

        return self.model_loader.get_memory(backend, model_tag)

    async def run_task(self, domain, payload):
        task_type = self.task_type

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
        elif task_type == "search_web": 
            result = domain.search_web(payload)
        elif task_type == "search_code": 
            result = domain.search_code(payload)

        if inspect.iscoroutine(result):
            result = await result

        return result

    async def unload_next(self, required_vram: int):
        """Busca o próximo modelo idle na prioridade e descarrega."""
        for task in self.unload_priority:
            if self.context.is_truly_idle(task):
                print(f"📦 [Orchestrator] Descarregando {task} para liberar espaço...")
                # Aqui chamamos o unload real do domínio
                domain = self.get_domain("gpu", self.payload.get("model"))
                domain.unload()
                self.context.garbage_collection()
                
                if self.context.get_free_vram() >= required_vram:
                    return True
        return False

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
        model_tag = payload.get("model")
        domain = self.get_domain("gpu", model_tag)

        async with self.context.orchestrator_lock:
            self.context.garbage_collection()
            
            if not domain.is_loaded():
                required = await domain.get_required_vram(payload)

                while self.context.get_free_vram() < required:
                    # Tenta descarregar o que já está parado
                    unloaded = await self.unload_next(required)
                    
                    if not unloaded:
                        # Se nada pode sair, espera alguém terminar de usar (mark_idle)
                        await self.wait_for_free_vram(required)
                
                domain.load()

            self.context.mark_busy(self.task_type, self)

        try:
            return await self.run_task(domain, payload)
        finally:
            # Libera o contador mas MANTÉM o modelo carregado ("quente")
            await self.context.mark_idle(self.task_type)
            self.context.garbage_collection()


class GpuOrchestrator:
    def __init__(self):
        # Passamos o resource_manager para o contexto gerenciar VRAM
        self.context = GpuContext()

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
        # Criamos o Runner. O Runner já possui a lógica de:
        # 1. Verificar VRAM
        # 2. Dar Unload em modelos idle se necessário
        # 3. Esperar se a VRAM estiver ocupada por processos busy
        # 4. Carregar o modelo
        # 5. Executar
        runner = Runner(task_type, payload, self.context)
        
        # Chamamos o runner.run que gerencia as Fases 1, 2 e 3
        # O retorno de runner.run é o que run_task devolve (pode ser valor ou gerador)
        try:
            result = await runner.run(payload)
            return result
        except Exception as e:
            print(f"❌ [GpuOrchestrator] Erro ao processar {task_type}: {str(e)}")
            raise e

gpu_orchestrator = GpuOrchestrator()