import asyncio
import threading
import time
import inspect
from queue import Queue, Empty
from typing import Any, AsyncGenerator, Union, Dict
import logging
from ..context import request_id_var

logger = logging.getLogger('JulyEngine.Orchestrators.GpuOrchestrator')

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

class GpuContext:
    def __init__(self):
        from ..resource_manager import resource_manager
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
            data = self.state.get(task_type)
            return data is not None and data.get("runner") is not None

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
        # Garante que passamos apenas o alias (string) para o loader, evitando erros de tipo no DB
        if isinstance(self.model, dict):
            model_tag = self.model.get("model_alias") or self.model.get("model")
        else:
            model_tag = self.model

        if self.task_type == "text_chat": 
            return self.model_loader.get_brain('gpu', model_tag)
        if self.task_type == "vision_chat": 
            return self.model_loader.get_eyes('gpu', model_tag)
        if self.task_type == "tts": 
            return self.model_loader.get_mouth('gpu', model_tag)
        if self.task_type == "stt": 
            return self.model_loader.get_ears('gpu', model_tag)
        if self.task_type in ["pix2pix", "image_generation", "image_resize"]:
            return self.model_loader.get_presence('gpu', model_tag)

        return self.model_loader.get_memory('gpu', model_tag)

    async def run_task(self, payload):
        task_type = self.task_type
        domain = self.get_domain()

        if task_type == "text_chat": 
            result = await domain.chat(payload)
        elif task_type == "vision_chat": 
            result = await domain.analyze(payload)
        elif task_type == "tts": 
            result = await domain.speak(payload)
        elif task_type == "stt": 
            result = await domain.listen(payload.get('audio'), payload.get('language'), payload)
        elif task_type in ["embedding", "embeddings"]: 
            result = await domain.embed(payload)
        elif task_type == "rag_add": 
            result = await domain.add_to_rag(payload.get("text"), payload.get("metadata"), payload.get("collection", "july_memory"))
        elif task_type == "rag_batch_add": 
            result = await domain.add_batch_to_rag(payload.get("documents", []), payload.get("collection", "july_memory"))
        elif task_type == "rag_search": 
            vector = payload.get("vector")
            if vector:
                result = await domain.search_with_details_vector(vector, payload.get("top_k", 3), payload.get("collection", "july_memory"))
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

    async def unload_next(self, required_vram: int):
        """Busca o próximo modelo idle na prioridade e descarrega."""
        for task in self.unload_priority:
            if self.context.is_loaded(task) and self.context.is_truly_idle(task):
                logger.info(f"📦 [Orchestrator] Descarregando {task} para liberar espaço...")
                runner = self.context.get_runner(task)
                await runner.unload()
                
                if self.context.get_free_vram() >= required_vram:
                    return True

        return False

    async def unload(self, timeout=60):
        elapsed_wait = 0
        while not self.context.is_truly_idle(self.task_type) and elapsed_wait < timeout:
            elapsed_wait += 1
            await asyncio.sleep(1)

        if elapsed_wait >= timeout:
            logger.warning(f"⚠️ [Orchestrator] Timeout aguardando {self.task_type} ficar idle. Forçando descarregamento.")
            # Se der timeout, tentamos descarregar mesmo assim ou falhamos
            # No caso de erro fatal, é melhor subir a exceção
            raise Exception(f"Timeout na GPU: {self.task_type} não está idle após {timeout}s.")

        # Garantir exclusividade no unload real
        with self.state_lock:
            self.get_domain().unload()
            self.context.garbage_collection()
            self.context.release_task(self.task_type)
            logger.info(f"✅ [Orchestrator] {self.task_type} descarregado com sucesso.")

    async def wait_for_free_vram(self, required_vram: int, timeout: int = 60):
        start_time = time.time()

        logger.debug(f"🚀 [Orchestrator] Aguardando por {required_vram} MB livres. Total Livre: {self.context.get_free_vram()} MB")

        async with self.context.condition:
            while self.context.get_free_vram() < required_vram:
                # Se algo que estava carregado ficou idle enquanto esperávamos, tentamos dar unload
                if any(self.context.is_loaded(t) and self.context.is_truly_idle(t) for t in self.context.state):
                    return 

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"VRAM insuficiente após {timeout}s.")

                try:
                    await asyncio.wait_for(self.context.condition.wait(), timeout=(timeout - elapsed))
                except asyncio.TimeoutError:
                    raise TimeoutError("Timeout na fila da GPU.")

    async def run(self, payload: dict):
        if isinstance(self.model, dict):
            model_alias = self.model.get("model_alias") or self.model.get("model")
        else:
            model_alias = self.model
            
        model_lock = self.context.get_model_lock(model_alias)

        async with model_lock:
            async with self.context.orchestrator_lock:
                self.context.garbage_collection()

                domain = self.get_domain()
                runner = self.context.get_runner(self.task_type)

                # Comparação robusta de modelos (por alias ou ID)
                model_changed = False
                if runner:
                    new_id = self.model.get("model_alias") or self.model.get("model") if isinstance(self.model, dict) else self.model
                    current_id = runner.model.get("model_alias") or runner.model.get("model") if isinstance(runner.model, dict) else runner.model
                    model_changed = current_id != new_id

                if self.context.is_loaded(self.task_type) and runner and model_changed:
                    await runner.unload()

                # 1. Garante que temos VRAM suficiente para rodar, mesmo se já estiver carregado
                # (Importante para modelos com offload dinâmico como Flux)
                required = await domain.get_required_vram(payload)
                
                while self.context.get_free_vram() < required:
                    # Tenta descarregar o que já está parado
                    unloaded = await self.unload_next(required)
                    
                    if not unloaded:
                        # Se não há mais nada para descarregar, tentamos reduzir camadas do próprio modelo (se GGUF)
                        if hasattr(domain, "decrement_layers"):
                            success = domain.decrement_layers()
                            if not success:
                                # Se chegou em 0 camadas e ainda não cabe, esperamos
                                await self.wait_for_free_vram(required, timeout=10)
                                if self.context.get_free_vram() < required:
                                    raise MemoryError(f"VRAM insuficiente para {self.task_type}. Requerido: {required}MB, Livre: {self.context.get_free_vram()}MB")
                            
                            # Atualiza o valor 'required' após o decremento
                            required = await domain.get_required_vram(payload)
                        else:
                            # Se o modelo não suporta decremento (ex: Flux), apenas espera por liberação externa
                            await self.wait_for_free_vram(required, timeout=10)
                            if self.context.get_free_vram() < required:
                                raise MemoryError(f"VRAM insuficiente para {self.task_type}. Requerido: {required}MB, Livre: {self.context.get_free_vram()}MB")

                if not domain.is_loaded():
                    domain.load()

                self.context.mark_busy(self.task_type, self)
        
        result = None
        is_stream = False
        try:
            result = await self.run_task(payload)
            if hasattr(result, '__aiter__'):
                is_stream = True
                
                async def generator_wrapper(gen, context, task_type):
                    try:
                        async for chunk in gen:
                            yield chunk
                    finally:
                        await context.mark_idle(task_type)
                        context.garbage_collection()
                
                return generator_wrapper(result, self.context, self.task_type)
            
            return result
        finally:
            if not is_stream:
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

        model_req = payload.get("model")
        if isinstance(model_req, str):
            # Tenta primeiro como modelo GGUF (banco de dados)
            model = model_service.get(model_req)
            
            # Se não for GGUF e for tarefa de texto, tenta resolver via Presets/Settings
            if not model and task_type in ["text_chat", "vision_chat"]:
                model = model_service.resolve_by_settings(model_req)
        else:
            model = model_req

        if not model:
            model = model_service.resolve_by_task_type(task_type)

        m_id = model.get('model_alias') or model.get('model') if isinstance(model, dict) else model
        slot = f"{task_type}_{m_id}"

        if slot not in self.slots:
            self.slots[slot] = Runner(task_type, model, self.context)

        try:
            result = await self.slots[slot].run(payload)
            return result
        except Exception as e:
            logger.error(f"❌ [GpuOrchestrator] Erro ao processar {task_type}: {str(e)}")
            raise e

    async def unload_model(self, model_alias: str):
        """
        Descarrega todas as instâncias de um modelo específico da GPU.
        Útil quando as configurações do modelo são alteradas e precisam ser recarregadas.
        """
        to_unload = []
        # Usamos list() para evitar erro de mudança de tamanho do dicionário durante a iteração
        for slot, runner in list(self.slots.items()):
            current_alias = runner.model.get("model_alias") or runner.model.get("model") if isinstance(runner.model, dict) else runner.model
            if current_alias == model_alias:
                to_unload.append((slot, runner))
        
        for slot, runner in to_unload:
            if self.context.is_loaded(runner.task_type):
                ctx_runner = self.context.get_runner(runner.task_type)
                # Verifica se o runner no contexto é de fato ESTE runner
                if ctx_runner == runner:
                    logger.info(f"📦 [Orchestrator] Descarregando modelo {model_alias} da tarefa {runner.task_type} devido a atualização de config.")
                    try:
                        await runner.unload()
                        
                        # Limpa também o cache do ModelLoader para forçar recarga de metadados
                        from ..model_loader import model_loader
                        model_loader.delete_instance('gpu', model_alias)
                        
                    except Exception as e:
                        logger.error(f"❌ [Orchestrator] Erro ao descarregar modelo {model_alias}: {e}")
            
            # Remove do cache de slots para que a próxima chamada crie um novo Runner com a config fresca
            if slot in self.slots:
                del self.slots[slot]

gpu_orchestrator = GpuOrchestrator()