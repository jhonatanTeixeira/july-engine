import asyncio
import os
import re
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from ..services.resource_calculator import estimate_vram_ram

logger = logging.getLogger("JulyEngine.Models.GGUF")

import re

def guess_num_layers(combined_name: str, params: float) -> int:
    """Adivinha o número de layers baseado no tamanho e arquitetura do modelo."""
    if not params or params == -1:
        return -1 # -1 significa "auto" para o llama.cpp
        
    combined_name = combined_name.lower()
    
    # ---------------------------------------------------------
    # DETECÇÃO DE MoE (Assinatura explícita ou notação oculta -aXb)
    # ---------------------------------------------------------
    is_moe = "mixtral" in combined_name or "moe" in combined_name or bool(re.search(r'-a(\d+(?:\.\d+)?)b', combined_name))
    
    if is_moe:
        # Mixtral 8x22B (~141B)
        if "mixtral" in combined_name and params >= 100:
            return 56 
        
        # Mixtral 8x7B (~47B)
        if "mixtral" in combined_name:
            return 32

        # Família Qwen MoE
        if "qwen" in combined_name:
            if params < 20: 
                return 24 # Ex: Qwen1.5-MoE-A2.7B (14.3B Total)
            if 25 <= params <= 35:
                return 32 # Ex: Qwen3-30B-A3B (Arquitetura otimizada)
            if 50 <= params <= 65:
                return 64 # Ex: Qwen2-57B-A14B
            return 32 # Fallback para MoEs desconhecidos

        # Família DeepSeek MoE
        if "deepseek" in combined_name:
            if params < 20:
                return 27 # DeepSeek-Coder-V2-Lite
            return 60     # DeepSeek-V2/V3 (Modelos gigantes)

        return 32 # Fallback genérico para MoE (arquitetura padrão tipo Mixtral)

    # ---------------------------------------------------------
    # MODELOS DENSOS (Sem MoE)
    # ---------------------------------------------------------
    
    # Família 7B - 9B
    if 7 <= params <= 9:
        if "gemma" in combined_name and params >= 9:
            return 42 # Gemma 2 9B
        return 32
        
    # Família 0.5B - 3B
    if params < 3:
        if "qwen" in combined_name:
            if params < 1: return 24 # Qwen 0.5B
            if 1 <= params <= 2: return 28 # Qwen 1.5B
            if 2 <= params <= 3: return 32 # Qwen 2.5/3B
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
        # Se chegou aqui, não é MoE (is_moe foi False)
        # Modelos densos nessa faixa (ex: Qwen2.5-32B) são muito profundos
        return 64 
        
    # Família 70B+
    if params >= 70:
        return 80

    return -1


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

    def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu": return 0
            
        meta = self.meta
        params_b = meta.get("num_params", 0)
        quant = meta.get("quantization", "Q4_K_M")
        combined_name = meta.get("model_id", "") + meta.get("filename", "")
        
        headers = payload.get("headers", {})
        effective_n_ctx = int(headers.get("x-context-window") or payload.get("n_ctx") or meta.get("context_window") or 2048)
        
        # O pulo do gato: 'layers' é o que queremos. 'physical_total' é o que o modelo TEM.
        layers_to_offload = payload.get("num_layers") or meta.get("num_layers") or -1
        physical_total = guess_num_layers(combined_name, params_b)

        estimates = estimate_vram_ram(
            combined_name,
            params_b, 
            quant, 
            effective_n_ctx, 
            layers=layers_to_offload,
            total_layers=physical_total # Agora o denominador está correto (32)
        )
        
        return int(estimates["estimated_vram_gb"] * 1024)

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
            from llama_cpp import Llama
            import llama_cpp
            
            logger.info(f"GGUF: Loading model {self.meta['model_alias']} on {self.backend} (n_ctx={effective_n_ctx})")
            
            params = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": effective_n_ctx,
                "verbose": False,
                "flash_attn": os.environ.get("FLASH_ATTN", "false").lower() == "true",
            }

            if quant := os.environ.get('KV_CACHE_QUANTIZATION', None):
                if quant == '8':
                    params["type_k"] = llama_cpp.GGML_TYPE_Q8_0
                    params["type_v"] = llama_cpp.GGML_TYPE_Q8_0
                if quant == '4':
                    params["type_k"] = llama_cpp.GGML_TYPE_Q4_0
                    params["type_v"] = llama_cpp.GGML_TYPE_Q4_0

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