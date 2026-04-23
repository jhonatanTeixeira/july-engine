import asyncio
import os
import re
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from ..services.resource_calculator import estimate_vram_ram, ModelMetadata

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

class GGUF:
    def __init__(self, backend, model):
        from huggingface_hub import hf_hub_download

        self.backend = backend
        self.meta = model
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.model = None
        self.model_path = hf_hub_download(repo_id=model["model_id"], filename=model["filename"])
        self.model_metadata = ModelMetadata(self.model_path)

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
        effective_n_ctx = int(headers.get("x-context-window") or payload.get("n_ctx") or meta.get("context_window") or 2048)
        
        # 2. Get layers config
        layers_to_offload = meta.get("num_layers") or -1
        
        # 3. Calculate with precision
        estimates = await estimate_vram_ram(
            self.model_path,
            context_window=effective_n_ctx,
            kv_cache_quantization=meta.get("kv_cache_quantization", "FP16"),
            gpu_layers=layers_to_offload
        )
        
        return estimates["total_vram_mb"]

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        from huggingface_hub import hf_hub_download

        meta = self.meta
        
        effective_n_ctx = n_ctx or meta.get("context_window") or int(os.environ.get("LLM_CTX_TOKENS", 2048))
        
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
            
            logger.info(f"GGUF: Loading model {self.meta['model_alias']} on {self.backend} (n_ctx={effective_n_ctx})")
            
            params = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": effective_n_ctx,
                "verbose": False,
            }
            logger.info(f"GGUF: Final params for Llama: {params}")

            if os.environ.get("FLASH_ATTN", "false").lower() == "true":
                params["flash_attn"] = True

            # Use KV Cache Quantization from metadata (preferred) or env var
            kv_quant = meta.get("kv_cache_quantization") or os.environ.get('KV_CACHE_QUANTIZATION')
            
            if kv_quant:
                kv_quant = str(kv_quant).upper()
                if "8" in kv_quant or "Q8_0" in kv_quant:
                    params["type_k"] = 8
                    params["type_v"] = 8
                    logger.info("GGUF: Using Q8_0 for KV Cache")
                elif "4" in kv_quant or "Q4_0" in kv_quant:
                    params["type_k"] = 4
                    params["type_v"] = 4
                    logger.info("GGUF: Using Q4_0 for KV Cache")
                else:
                    logger.info("GGUF: Using default FP16 for KV Cache")
            
            # Extração de Capacidades
            model_identifier = meta["model_id"] + meta["filename"]
            caps = detect_model_capabilities(model_identifier)

            if meta.get("template"):
                params["chat_format"] = meta["template"]
            else:
                params["chat_format"] = "jinja" if "jinja" in llama_cpp.llama_chat_format.CHAT_FORMATS else caps["chat_format"]
            
            if meta.get("model_type") == "vision" or caps["vision_handler"]:
                
                # Prepara o caminho do projetor se existir (modelos integrados como Qwen podem não usar)
                mmproj_path = None
                mmproj_id = meta.get("mmproj_id")
                mmproj_filename = meta.get("mmproj_filename")
                
                if mmproj_id and mmproj_filename:
                    mmproj_path = hf_hub_download(mmproj_id, mmproj_filename)

                v_handler = caps["vision_handler"]
                
                # Importa as classes avançadas do MTMDChatHandler
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
                        from llama_cpp.llama_chat_format import Qwen35ChatHandler
                        params["chat_handler"] = Qwen35ChatHandler(clip_model_path=mmproj_path) if mmproj_path else Qwen35ChatHandler()
                        
                    elif v_handler == "moondream":
                        from llama_cpp.llama_chat_format import MoondreamChatHandler
                        params["chat_handler"] = MoondreamChatHandler(clip_model_path=mmproj_path)
                        
                    elif v_handler == "llava-v1.6" or v_handler == "pixtral":
                        from llama_cpp.llama_chat_format import Llava16ChatHandler
                        params["chat_handler"] = Llava16ChatHandler(clip_model_path=mmproj_path)
                        
                    elif v_handler == "llava":
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)
                        
                except ImportError as exc:
                    logger.warning(f"GGUF: Handler {v_handler} não encontrado no teu llama-cpp-python. Fallback para Llava15. Erro: {exc}")
                    if mmproj_path:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)

            logger.info(f"GGUF: Final params for Llama: {params}")
            self.model = Llama(**params)
            
        except Exception as e:
            logger.error(f"GGUF: Failed to load {self.meta['model_alias']}: {e}")
            raise e

    def run_chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        headers = kwargs.pop("headers", {})
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

        response = self.model.create_chat_completion(
            messages,
            stream=stream,
            **kwargs
        )

        if stream:
            async def stream_adapter():
                tag_opened = force_reasoning
                buffer = ""
                raw_response = ""
                
                for chunk in response:
                    content = chunk["choices"][0].get("delta", {}).pop("content", "")
                    buffer += content
                    raw_response += content

                    if chunk["choices"][0].get("finish_reason") == "stop":
                        prompt_data = ""

                        for message in messages:
                            prompt_data += message["content"] if isinstance(message["content"], str) else "".join([c.get("text", "") for c in message['content']])

                        prompt_tokens = len(self.model.tokenize(prompt_data.encode('utf-8')))
                        completion_tokens = len(self.model.tokenize(raw_response.encode('utf-8')))

                        chunk["choices"][0].setdefault("usage", {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": (prompt_tokens + completion_tokens)
                        })

                    if '<' in buffer and re.match(r'(.*?)?<$|<\w+$', buffer) and not tag_opened:
                        continue

                    if '<' in buffer and re.match(r'(.*?)?<$|</$|</\w+$', buffer) and tag_opened:
                        continue

                    if '<think>' in buffer and not tag_opened:
                        tag_opened = True
                        before, after = buffer.split('<think>')

                        if before:
                            chunk["choices"][0]["delta"]["content"] = before
                            yield chunk
                        
                        if after:
                            chunk["choices"][0]["delta"]["reasoning_content"] = after
                            yield chunk

                        buffer = ""
                    
                    elif tag_opened and '</think>' in buffer:
                        tag_opened = False
                        before, after = buffer.split('</think>')

                        if before:
                            chunk["choices"][0]["delta"]["reasoning_content"] = before
                            yield chunk
                        
                        if after:
                            chunk["choices"][0]["delta"]["content"] = after
                            yield chunk
                        
                        buffer = ""
                    
                    elif tag_opened:
                        chunk["choices"][0]["delta"]["reasoning_content"] = content
                        yield chunk
                        buffer = ""

                    elif not tag_opened:
                        chunk["choices"][0]["delta"]["content"] = buffer
                        yield chunk
                        buffer = ""

                    await asyncio.sleep(0)

                # Flush residual do buffer se sobrar algo
                if buffer:
                    target_key = 'content' if not tag_opened else 'reasoning_content'
                    chunk["choices"][0]["delta"][target_key] = buffer
                    yield chunk

            return stream_adapter()
            
        else:
            # MODO NÃO-STREAM: Limpa a tag e separa tudo na raiz do JSON
            raw_content = response["choices"][0]["message"].get("content", "") or ""
            
            # Parser robusto para tags <think> mesmo não fechadas
            think_pattern = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
            match = think_pattern.search(raw_content)
            
            if match:
                reasoning = match.group(1).strip()
                # Remove o bloco de pensamento do conteúdo principal
                content = raw_content.replace(match.group(0), "").strip()
                
                # Se sobrar uma tag de fechamento solta (devido ao replace parcial do group 0)
                content = content.replace("</think>", "").strip()
                
                response["choices"][0]["message"]["reasoning_content"] = reasoning
                response["choices"][0]["message"]["content"] = content if content else None
            
            return response

    def unload(self, model_name: str):
        if self.model:
            # Deletar explicitamente para disparar o __del__ do C++
            del self.model
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