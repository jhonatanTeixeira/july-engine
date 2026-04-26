import asyncio
import os
import re
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from ..services.resource_calculator import estimate_vram_ram, ModelMetadata
from ..context import request_id_var, acquired_instances_var

logger = logging.getLogger("JulyEngine.Models.GGUF")

import re


def detect_model_capabilities(repo_id_or_filename: str) -> dict:
    """
    Usa RegEx para mapear o modelo para os Handlers específicos do fork JamePeng.
    """
    name = repo_id_or_filename.lower()
    capabilities = {
        "vision_handler": None,
        "chat_format": "jinja"
    }

    # ==========================================
    # 1. DETECÇÃO DO CHAT FORMAT FALLBACK
    # ==========================================
    if re.search(r"qwen[_\-\.]?(?:2\.5|3|4)", name):
        capabilities["chat_format"] = "chatml" 
    elif re.search(r"gemma[_\-\.]?[2-4]", name):
        capabilities["chat_format"] = "gemma"
    elif re.search(r"llama[_\-\.]?3|hermes", name):
        capabilities["chat_format"] = "llama-3"
    elif re.search(r"mistral|mixtral|pixtral|ministral", name):
        capabilities["chat_format"] = "mistral-instruct"

    # ==========================================
    # 2. DETECÇÃO DE VISÃO (JAMEPENG HANDLERS)
    # ==========================================
    if re.search(r"gemma[_\-\.]?4", name):
        capabilities["vision_handler"] = "gemma4"
    elif re.search(r"gemma[_\-\.]?3", name):
        capabilities["vision_handler"] = "gemma3"
    elif 'qwen' in name:
        if '2.5' in name and 'vl' in name:
            capabilities["vision_handler"] = "qwen25vl"
        if '3' in name and 'vl' in name:
            capabilities["vision_handler"] = "qwen3vl"
        if '3.5' in name:
            capabilities["vision_handler"] = "qwen35"
    elif re.search(r"pixtral|ministral", name):
        capabilities["vision_handler"] = "pixtral"
    elif re.search(r"moondream", name):
        capabilities["vision_handler"] = "moondream"
    elif re.search(r"llava[_\-\.]?v?1\.6", name):
        capabilities["vision_handler"] = "llava-v1.6"
    elif re.search(r"llava", name):
        capabilities["vision_handler"] = "llava"

    return capabilities

class SequencePool:
    def __init__(self, instances: List[Any]):
        self.instances = instances
        self._available = asyncio.Queue()
        for inst in instances:
            self._available.put_nowait(inst)

    async def acquire(self) -> Any:
        """Adquire uma instância livre do pool, com suporte a re-entrância por request_id."""
        rid = request_id_var.get()
        if rid:
            acquired = acquired_instances_var.get()
            if self in acquired:
                # Re-entrância: Esta request já reservou uma instância deste pool
                return acquired[self]
        
        # Caso contrário, espera por uma instância livre no pool
        inst = await self._available.get()
        
        if rid:
            # Reserva a instância para futuras chamadas nesta mesma request
            acquired = acquired_instances_var.get()
            acquired[self] = inst
            acquired_instances_var.set(acquired)
            
        return inst

    def release(self, inst: Any):
        """Libera a instância, a menos que esteja reservada para re-entrância."""
        rid = request_id_var.get()
        if rid:
            # Em requests HTTP rastreadas, não liberamos imediatamente pois o
            # segundo turno pode precisar da mesma instância.
            # A liberação real ocorrerá no Middleware ao fim da request.
            return
            
        self._real_release(inst)

    def _real_release(self, inst: Any):
        """Põe a instância de volta na fila de disponibilidade."""
        self._available.put_nowait(inst)

    def _force_release(self, inst: Any):
        """Força a liberação ignorando o request_id (usado pelo middleware)."""
        self._real_release(inst)

    def stop(self):
        """Nada a fazer para o pool simples."""
        pass

class GGUF:
    def __init__(self, backend, model):
        from huggingface_hub import hf_hub_download

        self.backend = backend
        self.meta = model
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.model = None
        self.model_path = hf_hub_download(repo_id=model["model_id"], filename=model["filename"])
        self.model_metadata = ModelMetadata(self.model_path)
        self.sequence_pool = None
        self.instances = []
        self.n_seq_max = int(model.get("n_seq_max") or model.get("n_parallel") or 1)
        self.offload_kqv = model.get("offload_kqv") if model.get("offload_kqv") is not None else True
        self.kv_unified = model.get("kv_unified") if model.get("kv_unified") is not None else True
        self.logits_all = model.get("logits_all") if model.get("logits_all") is not None else False

    def max_layers(self):
        return self.model_metadata.block_count

    def decrement_layers(self) -> bool:
        curr_layers = self.meta.get("num_layers")
        
        # Se for -1, resolvemos o total antes de decrementar
        if curr_layers == -1:
            curr_layers = self.max_layers()
        
        if curr_layers <= 0:
            logger.warning(f"GGUF: Model {self.meta['model_alias']} already at 0 layers. Cannot decrement further.")
            self.meta["num_layers"] = 0
            return False

        self.meta["num_layers"] = curr_layers - 1
        logger.info(f"GGUF: Decrementing layers for {self.meta['model_alias']}. New value: {self.meta['num_layers']}")
        return True

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu": 
            return 0
            
        meta = self.meta
        
        headers = payload.get("headers", {})
        n_ctx_per_req = int(headers.get("x-context-window") or payload.get("n_ctx") or meta.get("context_window") or 4096)
        
        # O contexto real na GPU é multiplicado pelo número de slots paralelos
        effective_n_ctx = n_ctx_per_req * self.n_seq_max
        
        # 2. Get layers config
        # Estima a VRAM necessária usando o calculador de recursos unificado
        estimate = await estimate_vram_ram(
            model_path=self.meta['file_path'],
            context_window=n_ctx_per_req,
            kv_cache_quantization=self.meta.get('kv_cache_quantization', 'FP16'),
            gpu_layers=meta.get("num_layers", -1),
            n_seq_max=self.n_seq_max,
            offload_kqv=self.offload_kqv,
            flash_attention=self.meta.get('flash_attn', True),
            logits_all=self.logits_all
        )
        
        return estimate["total_vram_mb"]

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        from huggingface_hub import hf_hub_download

        meta = self.meta
        
        # Aumentamos o padrão para 4096 para suportar agentes mais complexos
        n_ctx_per_req = n_ctx or meta.get("context_window") or int(os.environ.get("LLM_CTX_TOKENS", 4096))
        effective_n_ctx = n_ctx_per_req * self.n_seq_max
        
        if self.backend == 'cpu':
            n_gpu_layers = 0
        else:
            n_gpu_layers = num_layers if num_layers else meta.get("num_layers", -1)
            
        if self.is_loaded():
            if self.model.n_ctx() == effective_n_ctx:
                logger.debug(f"GGUF: Modelo {self.meta['model_alias']} já carregado. Reaproveitando!")
                return
            else:
                logger.info(f"GGUF: Reloading model {self.meta['model_alias']} because n_ctx changed ({self.model.n_ctx()} -> {effective_n_ctx})")
                self.unload(self.meta['model_alias'])
        
        model_path = self.model_path

        try:
            from llama_cpp import Llama
            import llama_cpp
            
            if self.backend == 'gpu':
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            logger.info(f"GGUF: Loading model {self.meta['model_alias']} on {self.backend} (n_seq_max={self.n_seq_max}, n_ctx_total={effective_n_ctx})")
            
            base_params = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx_per_req,
                "n_parallel": 1,
                "offload_kqv": self.offload_kqv,
                "kv_unified": self.kv_unified,
                "logits_all": self.logits_all,
                "use_mmap": True,
                "verbose": False,
            }
            logger.info(f"GGUF: Final base params for Llama instances: {base_params}")

            # Flash Attention: metadata > env var > default True
            flash_attn = meta.get("flash_attn")
            if flash_attn is None:
                flash_attn = os.environ.get("FLASH_ATTN", "true").lower() == "true"
            
            if flash_attn:
                base_params["flash_attn"] = True
                logger.info("GGUF: Flash Attention enabled")

            # Use KV Cache Quantization from metadata (preferred) or env var
            kv_quant = meta.get("kv_cache_quantization") or os.environ.get('KV_CACHE_QUANTIZATION')
            
            if kv_quant:
                kv_quant = str(kv_quant).upper()
                if "8" in kv_quant or "Q8_0" in kv_quant:
                    base_params["type_k"] = 8
                    base_params["type_v"] = 8
                    logger.info("GGUF: Using Q8_0 for KV Cache")
                elif "4" in kv_quant or "Q4_0" in kv_quant:
                    base_params["type_k"] = 2
                    base_params["type_v"] = 2
                    logger.info("GGUF: Using Q4_0 for KV Cache")
                else:
                    logger.info("GGUF: Using default FP16 for KV Cache")
            
            # Extração de Capacidades
            model_identifier = meta["model_id"] + meta["filename"]
            caps = detect_model_capabilities(model_identifier)

            if meta.get("template"):
                base_params["chat_format"] = meta["template"]
            else:
                base_params["chat_format"] = "jinja" if "jinja" in llama_cpp.llama_chat_format.CHAT_FORMATS else caps["chat_format"]

            # ==========================================
            # LOAD MULTIPLE INSTANCES
            # ==========================================
            self.instances = []
            for i in range(self.n_seq_max):
                logger.info(f"GGUF: Loading instance {i+1}/{self.n_seq_max} on {self.backend} (n_ctx={n_ctx_per_req})...")
                params = base_params.copy()

                # Cada instância precisa do seu próprio Chat Handler (estado independente)
                # Qwen Tool Calling & Reasoning Support (non-VL)
                if 'qwen' in model_identifier.lower() and not caps["vision_handler"]:
                    from .chat_handlers import QwenChatHandler
                    template = self.model_metadata.tokenizer_template or meta.get("template")
                    if isinstance(template, str) and template.strip():
                        params["chat_handler"] = QwenChatHandler(
                            template=template,
                            eos_token="<|im_end|>",
                            bos_token="<|im_start|>"
                        )
                        if "chat_format" in params:
                            del params["chat_format"]
                
                if meta.get("model_type") == "vision" or caps["vision_handler"]:
                    mmproj_path = None
                    mmproj_id = meta.get("mmproj_id")
                    mmproj_filename = meta.get("mmproj_filename")
                    if mmproj_id and mmproj_filename:
                        mmproj_path = hf_hub_download(mmproj_id, mmproj_filename)

                    v_handler = caps["vision_handler"]
                    try:
                        if v_handler == "gemma4":
                            from .chat_handlers import Gemma4Handler
                            params["chat_handler"] = Gemma4Handler(clip_model_path=mmproj_path) if mmproj_path else Gemma4Handler()
                        elif v_handler == "gemma3":
                            from llama_cpp.llama_chat_format import Gemma3ChatHandler
                            params["chat_handler"] = Gemma3ChatHandler(clip_model_path=mmproj_path) if mmproj_path else Gemma3ChatHandler()
                        elif v_handler == "qwen3vl":
                            from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                            params["chat_handler"] = Qwen3VLChatHandler(clip_model_path=mmproj_path) if mmproj_path else Qwen3VLChatHandler()
                        elif v_handler == "qwen25vl":
                            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                            params["chat_handler"] = Qwen25VLChatHandler(clip_model_path=mmproj_path) if mmproj_path else Qwen25VLChatHandler()
                        elif v_handler == "qwen35":
                            from .chat_handlers import Qwen35Handler
                            params["chat_handler"] = Qwen35Handler(clip_model_path=mmproj_path) if mmproj_path else Qwen35Handler()
                        elif v_handler == "moondream":
                            from llama_cpp.llama_chat_format import MoondreamChatHandler
                            params["chat_handler"] = MoondreamChatHandler(clip_model_path=mmproj_path)
                        elif v_handler == "llava-v1.6" or v_handler == "pixtral":
                            from llama_cpp.llama_chat_format import Llava16ChatHandler
                            params["chat_handler"] = Llava16ChatHandler(clip_model_path=mmproj_path)
                        elif v_handler == "llava":
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)
                    except ImportError:
                        if mmproj_path:
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)

                inst = Llama(**params)
                self.instances.append(inst)

            self.model = self.instances[0] # Usado para consultas de metadados
            self.sequence_pool = SequencePool(self.instances)
            
        except Exception as e:
            logger.error(f"GGUF: Failed to load {self.meta['model_alias']}: {e}")
            raise e

    async def run_chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        headers = kwargs.pop("headers", {})
        session_id = headers.get("x-session-id") or kwargs.pop("session_id", None)
        n_ctx = headers.get("x-context-window", None)
        
        if n_ctx:
            try:
                n_ctx = int(n_ctx)
            except ValueError:
                pass

        self.load(n_ctx, kwargs.pop('num_layers', None))

        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")
            
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.pop("reasoning_enabled", None)
        kwargs.pop("reasoning_effort", None)
        
        force_reasoning = self.meta.get("force_reasoning", False)

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = -1

        # O Truque do Force Reasoning: A IA começa já pensando
        if force_reasoning:
            messages.append({"role": "assistant", "content": "<think>\n"})

        # Adquire uma instância do pool
        inst = await self.sequence_pool.acquire()
        
        # Sempre reseta para garantir que não há lixo no KV Cache
        try:
            # inst.reset()
            logger.debug(f"GGUF: Instance KV cache reset")
        except Exception as e:
            logger.warning(f"GGUF: Could not reset instance: {e}")

        try:
            response = inst.create_chat_completion(
                messages,
                stream=stream,
                **kwargs
            )
        except Exception as e:
            self.sequence_pool.release(inst)
            raise e

        if stream:
            async def stream_adapter():
                try:
                    tag_opened = force_reasoning
                    buffer = ""
                    raw_response = ""
                    
                    for chunk in response:
                        delta = chunk["choices"][0].get("delta", {})
                        
                        finish_reason = chunk["choices"][0].get("finish_reason")
                        if finish_reason in ["stop", "tool_calls"]:
                            prompt_data = ""

                            for message in messages:
                                # prompt_data += message["content"] if isinstance(message["content"], str) else "".join([c.get("text", "") for c in message['content']])
                                prompt_data += "\n".join(c['text'] for c in message.get("content")) if isinstance(message.get("content"), list) else (message.get("content") or "")

                            prompt_tokens = len(inst.tokenize(prompt_data.encode('utf-8')))
                            completion_tokens = len(inst.tokenize(raw_response.encode('utf-8')))

                            chunk["choices"][0].setdefault("usage", {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": (prompt_tokens + completion_tokens)
                            })

                            yield chunk
                            continue

                        # Pass-through para outros dados que não sejam texto puro
                        if "audio_url" in delta or "image_url" in delta or "tool_calls" in delta:
                            yield chunk
                            continue

                        content = delta.pop("content", "")
                        buffer += content
                        raw_response += content
                        for tag_start, tag_end in [("<think>", "</think>"), ("<thought>", "</thought>"), ("<|thought|>", "<|thought|>"), ("<|channel>thought", "<channel|>")]:
                            if tag_start in buffer:
                                parts = buffer.split(tag_start, 1)
                                
                                # Se tiver texto ANTES da tag, entrega como content normal
                                if parts[0]:
                                    chunk["choices"][0]["delta"]["content"] = parts[0]
                                    yield chunk
                                
                                buffer = parts[1]
                                tag_opened = True
                                break

                            if tag_end in buffer and tag_opened:
                                parts = buffer.split(tag_end, 1)
                                if parts[0]:
                                    chunk["choices"][0]["delta"]["reasoning_content"] = parts[0]
                                    yield chunk
                                
                                buffer = parts[1]
                                tag_opened = False
                                break

                        # Entrega o buffer se atingir um tamanho razoável ou quebra de linha
                        if len(buffer) > 4 or "\n" in buffer:
                            target_key = 'content' if not tag_opened else 'reasoning_content'
                            chunk["choices"][0]["delta"][target_key] = buffer
                            buffer = ""
                            yield chunk

                        await asyncio.sleep(0)

                    # Flush residual do buffer se sobrar algo
                    if buffer:
                        target_key = 'content' if not tag_opened else 'reasoning_content'
                        chunk["choices"][0]["delta"][target_key] = buffer
                        yield chunk
                finally:
                    # Libera a instância ao fim do stream
                    self.sequence_pool.release(inst)
                    logger.debug(f"GGUF: Instance released back to pool")

            return stream_adapter()
            
        else:
            try:
                # MODO NÃO-STREAM: Limpa a tag e separa tudo na raiz do JSON
                raw_content = response["choices"][0]["message"].get("content", "") or ""

                if force_reasoning:
                    raw_content = "<think>" + raw_content
                
                # Parser robusto para tags <think>, <thought> ou <|thought|> mesmo não fechadas
                think_pattern = re.compile(r"<(?:\|thought\||think|thought)>(.*?)(?:</(?:think|thought)>|(?=<\|)|$)", re.DOTALL)
                match = think_pattern.search(raw_content)
                
                if match:
                    reasoning = match.group(1).strip()
                    # Remove o bloco de pensamento do conteúdo principal
                    # Usamos match.group(0) para remover a tag inteira
                    content = raw_content.replace(match.group(0), "").strip()
                    
                    # Cleanup de tags de fechamento remanescentes se necessário
                    for close_tag in ["</think>", "</thought>", "<channel|>"]:
                        content = content.replace(close_tag, "").strip()
                    
                    response["choices"][0]["message"]["reasoning_content"] = reasoning
                    response["choices"][0]["message"]["content"] = content if content else None
                
                return response
            finally:
                # Libera a instância após o processamento não-stream
                self.sequence_pool.release(inst)
                logger.debug(f"GGUF: Instance released back to pool")

    def unload(self, model_name: str):
        if self.instances:
            # Para o pool e limpa todas as instâncias
            if self.sequence_pool:
                self.sequence_pool.stop()
            
            for inst in self.instances:
                del inst
            
            self.instances = []
            self.model = None
            
        import gc
        gc.collect()
        
        # Se estivermos usando CUDA, tenta esvaziar o cache via torch se disponível
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
            
        logger.info(f"GGUF: Unloaded {model_name} and cleared CUDA cache")

    def is_loaded(self):
        return self.model is not None