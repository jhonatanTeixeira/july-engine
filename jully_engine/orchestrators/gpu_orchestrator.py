import os
import asyncio
import logging
import threading
import queue
import inspect
import time
from typing import Any, Dict, Optional, Union, AsyncGenerator

from ..resource_manager import resource_manager
from ..routers.calculator import estimate_vram_ram
from ..routers.models import load_models_db
from fastapi import HTTPException

logger = logging.getLogger("JulyEngine.Orchestrators.GpuOrchestrator")

# ---------------------------------------------------------------------------
# 1. HEURÍSTICA DE LAYERS
# ---------------------------------------------------------------------------

def guess_num_layers(combined_name: str, params: float) -> int:
    """Adivinha o número de layers baseado no tamanho do modelo."""
    if not params or params == -1:
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


# ---------------------------------------------------------------------------
# 2. O WORKER DA GPU (Thread Dedicada e Eterna)
# ---------------------------------------------------------------------------

def gpu_thread_worker(task_type: str, in_q: queue.Queue, out_q: queue.Queue, ready_event: threading.Event):
    """
    Thread isolada e eterna. 
    Mantém o contexto CUDA vivo. Lê da in_q, processa e joga na out_q.
    """
    logger.info(f">> [GpuThread-{task_type}] Iniciada (Thread ID: {threading.get_native_id()})")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    domain_instance = None
    ready_event.set()

    while True:
        try:
            msg = in_q.get() 
            cmd = msg.get("cmd")
            payload = msg.get("payload", {})

            if cmd == "SHUTDOWN":
                logger.info(f">> [GpuThread-{task_type}] Desligando...")
                in_q.task_done()
                break

            elif cmd == "LOAD":
                # BLINDAGEM 3: Se já tem um modelo, descarrega forçadamente antes de subir o novo
                if domain_instance is not None:
                    try:
                        if hasattr(domain_instance, 'unload'): domain_instance.unload()
                        elif hasattr(domain_instance, '_strategy') and hasattr(domain_instance._strategy, 'clear'):
                            domain_instance._strategy.clear()
                    except Exception: pass
                    finally:
                        domain_instance = None
                        import gc; gc.collect()
                        from ..resource_manager import resource_manager
                        resource_manager.clear_memory()

                model_tag = payload.get("model")
                backend = "gpu"
                from ..model_loader import model_loader
                
                if task_type == "text_chat": domain_instance = model_loader.get_brain(backend, model_tag)
                elif task_type == "vision_chat": domain_instance = model_loader.get_eyes(backend, model_tag)
                elif task_type == "tts": domain_instance = model_loader.get_mouth(backend, model_tag)
                elif task_type == "stt": domain_instance = model_loader.get_ears(backend, model_tag)
                elif task_type in ["embedding", "rag_add", "rag_batch_add", "rag_search", "rag_vector_add", "rag_search_details", "rag_update"]: 
                    domain_instance = model_loader.get_memory(backend, model_tag)
                elif task_type in ["pix2pix", "image_generation"]: domain_instance = model_loader.get_presence(backend, model_tag)
                
                out_q.put({"status": "LOAD_OK"})

            elif cmd == "UNLOAD":
                # BLINDAGEM 4: Try/Finally para garantir que a VRAM seja limpa mesmo se a strategy chorar
                try:
                    if hasattr(domain_instance, 'unload'):
                        domain_instance.unload()
                    elif hasattr(domain_instance, '_strategy'):
                        strategy = domain_instance._strategy
                        if hasattr(strategy, 'unload'):
                            sig = inspect.signature(strategy.unload)
                            if len(sig.parameters) > 0: strategy.unload(domain_instance.model_tag)
                            else: strategy.unload()
                        elif hasattr(strategy, 'clear'):
                            strategy.clear()
                except Exception as e:
                    logger.warning(f"Aviso no UNLOAD do {task_type}: {e}")
                finally:
                    domain_instance = None
                    import gc; gc.collect()
                    from ..resource_manager import resource_manager
                    resource_manager.clear_memory()
                    out_q.put({"status": "UNLOAD_OK"})

            elif cmd == "RUN_TASK":
                if not domain_instance:
                    out_q.put({"type": "ERROR", "data": "Domain instance not loaded!"})
                    in_q.task_done()
                    continue

                if task_type == "text_chat": result = domain_instance.chat(payload)
                elif task_type == "vision_chat": result = domain_instance.analyze(payload)
                elif task_type == "tts": result = domain_instance.speak(payload)
                elif task_type == "stt": result = domain_instance.listen(payload.get('audio'), payload.get('language'), payload)
                elif task_type == "embedding": result = domain_instance.embed(payload)
                elif task_type == "rag_add": result = domain_instance.add_to_rag(payload.get("text"), payload.get("metadata"), payload.get("collection", "july_memory"))
                elif task_type == "rag_batch_add": result = domain_instance.add_batch_to_rag(payload.get("documents", []), payload.get("collection", "july_memory"))
                elif task_type == "rag_search": result = domain_instance.search(payload.get("query"), payload.get("top_k", 3), payload.get("collection", "july_memory"))
                elif task_type == "rag_vector_add": result = domain_instance.add_vector_to_rag(payload.get("vector"), payload.get("text", ""), payload.get("metadata"), payload.get("collection", "july_memory"))
                elif task_type == "rag_search_details": 
                    vector = payload.get("vector")
                    if vector:
                        result = domain_instance.search_with_details_vector(vector, payload.get("top_k", 3), payload.get("collection", "july_memory"))
                    else:
                        # Fallback se for texto (emb -> search)
                        emb = asyncio.run(domain_instance.embed({"input": payload.get("query")}))
                        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list): emb = emb[0]
                        result = domain_instance.search_with_details_vector(emb, payload.get("top_k", 3), payload.get("collection", "july_memory"))
                elif task_type == "rag_update": result = domain_instance.update_embedding(str(payload.get("id")), payload.get("vector"))
                elif task_type in ["pix2pix", "image_generation"]: result = domain_instance.generate(payload)

                if inspect.iscoroutine(result):
                    result = asyncio.run(result)

                if inspect.isasyncgen(result):
                    async def consume_async_stream(async_gen):
                        async for chunk in async_gen:
                            out_q.put({"type": "CHUNK", "data": chunk})
                    asyncio.run(consume_async_stream(result))
                    out_q.put({"type": "DONE", "data": None})

                elif inspect.isgenerator(result) or hasattr(result, '__next__'):
                    for chunk in result:
                        out_q.put({"type": "CHUNK", "data": chunk})
                    out_q.put({"type": "DONE", "data": None})

                else:
                    out_q.put({"type": "DONE", "data": result})
                    
                # BLINDAGEM 2: Matar a referência fantasma no final da task
                if 'result' in locals(): del result

        except Exception as e:
            logger.exception(f"GpuThread[{task_type}] Erro fatal de execução: {e}")
            out_q.put({"type": "ERROR", "data": str(e)})
        finally:
            if 'msg' in locals() and msg.get("cmd") != "SHUTDOWN":
                in_q.task_done()

# ---------------------------------------------------------------------------
# 3. O ORQUESTRADOR CENTRAL (Gestor de Threads)
# ---------------------------------------------------------------------------

class GpuOrchestrator:
    """
    Manages GPU-bound tasks using dedicated long-running threads and resource management.
    Controls model life-cycle based on VRAM constraints.
    """
    def __init__(self):
        self.running = False
        self.workers = {}
        self.lock = threading.Lock()
        
        self.priorities = ["pix2pix", 'tts', 'stt', 'vision', 'memory', 'llm']
        self.busy_counts = {k: 0 for k in ["llm", "vision", "tts", "stt", "pix2pix", "memory"]}
        self.conditions = {k: threading.Condition(self.lock) for k in self.busy_counts.keys()}
        self.active_gpu_models: Dict[str, str] = {} 
        
        # Mapa correto: De Task para Chave de Memória
        self.task_to_key = {
            "text_chat": "llm",
            "vision_chat": "vision",
            "tts": "tts",
            "stt": "stt",
            "embedding": "memory",
            "rag_add": "memory",
            "rag_batch_add": "memory",
            "rag_search": "memory",
            "rag_vector_add": "memory",
            "rag_search_details": "memory",
            "rag_update": "memory",
            "pix2pix": "pix2pix",
            "image_generation": "pix2pix"
        }
        
        # Mapa reverso: Da Chave de Memória para o Canal do Pipe/Fila
        self.key_to_task = {
            "llm": "text_chat",
            "vision": "vision_chat",
            "tts": "tts",
            "stt": "stt",
            "pix2pix": "pix2pix",
            "memory": "embedding"
        }
        
    async def start(self):
        if not self.running:
            with self.lock:
                self.running = True
                task_types = [
                    "text_chat", "vision_chat", "stt", "tts", "embedding", 
                    "rag_add", "rag_batch_add", "rag_search", "rag_vector_add", "rag_search_details", "rag_update",
                    "pix2pix", "image_generation"
                ]
                for tt in task_types:
                    if tt == "image_generation": 
                        continue 
                        
                    in_q = queue.Queue()
                    out_q = queue.Queue()
                    ready_event = threading.Event()
                    
                    t = threading.Thread(
                        target=gpu_thread_worker, 
                        args=(tt, in_q, out_q, ready_event),
                        name=f"JulyThread-{tt}",
                        daemon=True
                    )
                    t.start()
                    ready_event.wait(timeout=5) 
                    
                    self.workers[tt] = {"thread": t, "in": in_q, "out": out_q}
                    
                # Compartilha a thread do pix2pix com image_generation
                self.workers["image_generation"] = self.workers["pix2pix"]
                
            logger.info("GpuOrchestrator: Long-running Threads Iniciadas (RAM Windows Protegida).")

    async def stop(self):
        with self.lock:
            self.running = False
            for tt, w in self.workers.items():
                if tt == "image_generation": continue
                if w["thread"].is_alive():
                    w["in"].put({"cmd": "SHUTDOWN"})
                    w["thread"].join(timeout=2)
            self.workers.clear()
            self.active_gpu_models.clear()

    # --- CONTROLES DE ESTADO ---
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

    # --- UNLOAD VIA QUEUE ---
    async def _unload_worker(self, model_key: str):
        task_type = self.key_to_task.get(model_key)
        if task_type and task_type in self.workers:
            logger.info(f"GpuOrchestrator: Enviando UNLOAD para {model_key}...")
            out_q = self.workers[task_type]["out"]
            
            # Limpa qualquer lixo da fila de saída antes de mandar
            while not out_q.empty(): 
                try: out_q.get_nowait()
                except queue.Empty: break
            
            self.workers[task_type]["in"].put({"cmd": "UNLOAD"})
            await asyncio.to_thread(out_q.get) # Aguarda confirmação OK
            self.active_gpu_models.pop(model_key, None)

    # --- O CORAÇÃO DO GERENCIADOR DE RECURSOS ---
    async def ensure_resources(self, model_key: str, required_vram: float = 4000) -> float:
        """
        Unloads models based on priority to free up VRAM.
        Returns the final available VRAM.
        """
        available = resource_manager.get_available_vram_mb()
        if available >= required_vram: 
            return available
            
        busy_candidates = []
        
        for candidate in self.priorities:
            if candidate not in self.active_gpu_models: 
                continue
            
            with self.lock:
                is_busy = self.busy_counts.get(candidate, 0) > 0
                
            if is_busy:
                busy_candidates.append(candidate)
                continue
                
            await self._unload_worker(candidate)
            available = resource_manager.get_available_vram_mb()
            if available >= required_vram: 
                return available

        for busy in busy_candidates:
            logger.warning(f"GpuOrchestrator: Timeout waiting for {busy}, forcing unload.")
            with self.lock: 
                condition = self.conditions[busy]
            
            def wait_cond():
                with self.lock: 
                    return condition.wait(timeout=300)
                    
            woke_up = await asyncio.to_thread(wait_cond)
            if not woke_up: 
                logger.warning(f"Timeout em {busy}. Forçando Unload!")
            
            await self._unload_worker(busy)
            if (available := resource_manager.get_available_vram_mb()) >= required_vram: 
                return available
                
        return available

    # --- O PONTO DE ENTRADA DO BRIDGE ---
    async def submit_task(self, task_type: str, payload: Any) -> Union[Any, AsyncGenerator[Any, None]]:
        if not self.running: raise RuntimeError("GpuOrchestrator not running")
        if task_type not in self.workers: raise ValueError(f"Unknown GPU task type: {task_type}")
        
        model_tag = payload.get("model")
        model_key = self.task_to_key.get(task_type, "llm")

        # 1. Resource pre-check and unloading
        db = load_models_db()
        meta = db.get(model_tag, {})

        effective_n_ctx = 2048
        n_layers = -1
        params_b = 0
        quant = "Q4_K_M"

        # Heurística para modelos específicos
        if model_tag in ["fastvlm", "moondream"]:
            required_vram_mb = 2048 # ~2GB fixo para esses modelos
        elif model_tag == 'lcm':
            required_vram_mb = 200 # Pico de ativação do SD1.5 (VAE Decode)
        elif meta.get('model_type', 'text') in ['text', 'vision']:
            params_b = meta.get("num_params", 0)
            quant = meta.get("quantization", "Q4_K_M")
            
            headers = payload.get("headers", {})
            effective_n_ctx = int(headers.get("x-context-window") or meta.get("context_window") or 2048)
            
            n_layers = meta.get("num_layers", -1)
            if n_layers == -1:
                n_layers = guess_num_layers(model_tag + meta.get("filename", ""), params_b)
            
            estimates = estimate_vram_ram(params_b, quant, effective_n_ctx, n_layers)
            required_vram_mb = estimates["estimated_vram_gb"] * 1024
        else:
            required_vram_mb = meta.get('estimated_vram', 0)
            
        if self.active_gpu_models.get(model_key) == model_tag:
            required_vram_mb = 0

        # 2. Try to free up memory
        available_vram = await self.ensure_resources(model_key, required_vram_mb)

        # 3. Iterative layer optimization se ainda não couber (apenas para GGUF)
        if meta.get('model_type', 'text') in ['text', 'vision'] and model_tag not in ["fastvlm", "moondream"] and available_vram < required_vram_mb and n_layers > 0:
            logger.info(f"GpuOrchestrator: Model {model_tag} ({required_vram_mb:.2f}MB) too big for VRAM ({available_vram:.2f}MB). Decrementing layers...")
            
            while n_layers > 0 and available_vram < required_vram_mb:
                n_layers -= 1
                estimates = estimate_vram_ram(params_b, quant, effective_n_ctx, n_layers)
                required_vram_mb = estimates["estimated_vram_gb"] * 1024
            
            logger.info(f"GpuOrchestrator: Optimized model to {n_layers} layers ({required_vram_mb:.2f}MB required)")
            
            # Safety margin: decrement one more if possible
            if n_layers > 0:
                n_layers -= 1
                
            payload["num_layers"] = n_layers

        if available_vram < required_vram_mb:
            logger.error(f"GpuOrchestrator: Insufficient VRAM for {model_tag}. Required: {required_vram_mb:.2f}MB, Available: {available_vram:.2f}MB.")
            raise HTTPException(status_code=422, detail=f"Insufficient VRAM even with 0 layers. Required: {required_vram_mb:.2f}MB, Available: {available_vram:.2f}MB.")

        self.mark_busy(model_key)
        
        try:
            in_q = self.workers[task_type]["in"]
            out_q = self.workers[task_type]["out"]

            # Limpa lixos residuais da fila de resposta por garantia
            while not out_q.empty(): 
                try: out_q.get_nowait()
                except queue.Empty: break

            if self.active_gpu_models.get(model_key) != model_tag:
                logger.info(f"GpuOrchestrator: Enviando LOAD {model_tag} para {task_type}")
                in_q.put({"cmd": "LOAD", "payload": payload})
                
                load_resp = await asyncio.to_thread(out_q.get)
                if load_resp.get("type") == "ERROR":
                    raise RuntimeError(f"Falha no load: {load_resp.get('data')}")
                    
                self.active_gpu_models[model_key] = model_tag

            logger.info(f"GpuOrchestrator: Executing {task_type} with tag {model_tag}")
            in_q.put({"cmd": "RUN_TASK", "payload": payload})

            # Aguarda a primeira mensagem para definir o tipo de retorno
            msg = await asyncio.to_thread(out_q.get)
            m_type = msg.get("type")

            if m_type == "ERROR":
                raise HTTPException(status_code=500, detail=msg.get("data"))
                
            elif m_type == "DONE":
                self.mark_idle(model_key)
                resource_manager.clear_memory()
                return msg.get("data")
                
            elif m_type == "CHUNK":
                return self._consume_stream(out_q, first_chunk=msg.get("data"), model_key=model_key)

        except Exception as e:
            self.mark_idle(model_key)
            logger.exception(f"Erro inesperado no submit_task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _consume_stream(self, out_q: queue.Queue, first_chunk: Any, model_key: str) -> AsyncGenerator[Any, None]:
        try:
            yield first_chunk
            while True:
                msg = await asyncio.to_thread(out_q.get)
                m_type = msg.get("type")
                if m_type == "DONE":
                    break
                elif m_type == "ERROR":
                    raise Exception(msg.get("data"))
                elif m_type == "CHUNK":
                    yield msg.get("data")
        finally:
            self.mark_idle(model_key)

gpu_orchestrator = GpuOrchestrator()