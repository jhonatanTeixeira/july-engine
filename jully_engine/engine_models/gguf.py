import asyncio
import os
import re
import json
import logging
import time
from typing import Any, Dict, List, Optional
import uuid
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

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
        n_gpu_layers = num_layers if num_layers else (meta.get("num_layers", -1) if self.backend == "gpu" else 0)
            
        if self.is_loaded():
            if self.model.n_ctx() == effective_n_ctx:
                logger.debug(f"GGUF: Modelo {self.meta['model_alias']} já carregado. Reaproveitando!")
                return

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
                in_reasoning = force_reasoning
                buffer = ''
                
                for chunk in response:
                    raw_text = chunk["choices"][0].get("delta", {}).get("content", "")
                    
                    if not raw_text:
                        yield chunk
                        continue
                    
                    buffer += raw_text
                    
                    if re.search('<\/?(\w+)?$', buffer):
                        continue
                    
                    if '<think>' in buffer:
                        in_reasoning=True
                        raw_text = buffer.replace('<think>', '')
                        buffer = ''
                    
                    if '</think>' in buffer:
                        in_reasoning=False
                        raw_text = raw_text.rsplit('>')[1]
                        buffer = ''

                    mod_chunk = dict(chunk)
                    
                    if in_reasoning:
                        mod_chunk["choices"][0]["delta"] = {"reasoning_content": raw_text}
                    else:
                        mod_chunk["choices"][0]["delta"] = {"content": raw_text}
                        
                    yield mod_chunk
                    
                    buffer = ''
                    
                    await asyncio.sleep(0)
            return stream_adapter()
            
        else:
            # MODO NÃO-STREAM: Limpa a tag e separa tudo na raiz do JSON
            raw_content = response["choices"][0]["message"].get("content", "")
            
            # Remove a tag <think> do final que o modelo pode ter esquecido de fechar
            if force_reasoning and not "</think>" in raw_content:
                raw_content += "</think>"
                
            # Se forçou, já sabemos que começa com o pensamento (o parser C++ do llama mescla)
            # Mas vamos usar Regex para capturar tudo de forma segura:
            pattern = re.compile(r"<(?:think|thought|reasoning)>(.*?)</(?:think|thought|reasoning)>", re.DOTALL)
            match = pattern.search(raw_content)
            
            if match:
                reasoning = match.group(1).strip()
                content = (raw_content[:match.start()] + raw_content[match.end():]).strip()
                response["choices"][0]["message"]["reasoning_content"] = reasoning
                response["choices"][0]["message"]["content"] = content if content else None
            
            return response

    def unload(self, model_name: str):
        self.model = None
        import gc
        gc.collect()
        logger.info(f"GGUF: Unloaded {model_name}")

    def is_loaded(self):
        return self.model is not None