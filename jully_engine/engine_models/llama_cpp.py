import asyncio
import os
import re
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("JulyEngine.Models.GGUF")

def detect_model_type(repo_id_or_filename: str) -> str:
    name = repo_id_or_filename.lower()
    if "moondream" in name:
        return "moondream"
    if "nanollava" in name:
        return "nanollava"
    if "llava-v1.6" in name or "mistral-7b-instruct-v0.2" in name:
        return "llava-v1.6"
    return "llava"

class GGUF:
    def __init__(self, backend, model):
        self.backend = backend
        self.meta = model
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.model = None

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
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
        
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=meta["model_id"], filename=meta["filename"])

        try:
            logger.info(f"GGUF: Loading model {self.meta['model_alias']} on {self.backend} (n_ctx={effective_n_ctx})")
            
            params = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": effective_n_ctx,
                "verbose": False
            }

            if meta.get("template"):
                params["chat_format"] = meta["template"]

            if meta.get("model_type") == "vision":
                mmproj_id = meta.get("mmproj_id")
                mmproj_filename = meta.get("mmproj_filename")
                if not mmproj_id or not mmproj_filename:
                    raise ValueError(f"GGUF: Vision model {self.meta['model_alias']} missing mmproj.")
                mmproj_path = hf_hub_download(mmproj_id, mmproj_filename)
                
                vision_type = detect_model_type(meta["model_id"] + meta["filename"])
                
                if vision_type == "moondream":
                    from llama_cpp.llama_chat_format import MoondreamChatHandler
                    params["chat_handler"] = MoondreamChatHandler(clip_model_path=mmproj_path)
                elif vision_type == "nanollava":
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                    params["chat_handler"] = NanoLlavaChatHandler(clip_model_path=mmproj_path)
                else:
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)

            from llama_cpp import Llama
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
            # Verifica se já não existe uma mensagem do sistema ou se o modelo já suporta nativamente
            # Para o nosso caso, apenas concatenamos se a última não for assistente começando com think
            messages.append({"role": "assistant", "content": "<think>\n"})

        response = self.model.create_chat_completion(
            messages,
            stream=stream,
            **kwargs
        )

        if stream:
            async def stream_adapter():
                in_reasoning = force_reasoning
                buffer = ''
                
                for chunk in response:
                    delta = chunk["choices"][0].get("delta", {})
                    raw_text = delta.get("content", "")
                    
                    if not raw_text:
                        yield chunk
                        continue
                    
                    # Lógica simplificada de detecção de tags
                    if "<think>" in raw_text:
                        in_reasoning = True
                        raw_text = raw_text.replace("<think>", "")
                    
                    if "</think>" in raw_text:
                        in_reasoning = False
                        # O que vem antes do fechamento é pensamento, o que vem depois é conteúdo
                        parts = raw_text.split("</think>")
                        
                        # Emite a parte do pensamento
                        if parts[0]:
                            mod_chunk = dict(chunk)
                            mod_chunk["choices"][0]["delta"] = {"reasoning_content": parts[0]}
                            yield mod_chunk
                        
                        # Continua com a parte do conteúdo
                        raw_text = parts[1]
                        if not raw_text: continue

                    mod_chunk = dict(chunk)
                    if in_reasoning:
                        mod_chunk["choices"][0]["delta"] = {"reasoning_content": raw_text}
                    else:
                        mod_chunk["choices"][0]["delta"] = {"content": raw_text}
                        
                    yield mod_chunk
                    await asyncio.sleep(0)
            return stream_adapter()
            
        else:
            # MODO NÃO-STREAM: Limpa a tag e separa tudo na raiz do JSON
            raw_content = response["choices"][0]["message"].get("content", "")
            
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