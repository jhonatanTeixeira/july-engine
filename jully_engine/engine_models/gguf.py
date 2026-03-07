import os
import json
import logging
import time
from typing import Any, Dict, List, Optional
import uuid
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

logger = logging.getLogger("JulyEngine.Models.GGUF")

def detect_model_type(repo_id_or_filename: str) -> str:
    name = repo_id_or_filename.lower()
    if "moondream" in name:
        return "moondream"
    if "nanollava" in name:
        return "nanollava"
    if "llava-v1.6" in name or "mistral-7b-instruct-v0.2" in name: # Exemplo de LLaVA 1.6
        return "llava-v1.6"
    return "llava"

class GGUF:
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.active_models = {}
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.models_json_path = os.path.join(self.cache_dir, "july_models.json")

    def _get_model_metadata(self, model_alias: str) -> Dict[str, Any]:
        if not os.path.exists(self.models_json_path):
            raise FileNotFoundError(f"GGUF: {self.models_json_path} not found. Please download models via API first.")
            
        with open(self.models_json_path, "r", encoding="utf-8") as f:
            db = json.load(f)
            
        if model_alias not in db:
            raise ValueError(f"GGUF: Model alias '{model_alias}' not found in registry.")
            
        return db[model_alias]

    def _ensure_downloaded(self, repo_id: str, filename: str) -> str:
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        except LocalEntryNotFoundError:
            logger.info(f"GGUF: Downloading {repo_id}/{filename}")
            return hf_hub_download(repo_id=repo_id, filename=filename)
    
    def load(self, model_alias: str, n_ctx: Optional[int] = None):
        meta = self._get_model_metadata(model_alias)
        
        # Determine effective context window
        effective_n_ctx = n_ctx
        if effective_n_ctx is None:
            effective_n_ctx = meta.get("context_window")
            if effective_n_ctx is None:
                 effective_n_ctx = int(os.environ.get("LLM_CTX_TOKENS", 2048))

        # Ensures model is downloaded
        model_path = self._ensure_downloaded(meta["model_id"], meta["filename"])

        if model_alias in self.active_models:
            current_model = self.active_models[model_alias]
            if current_model.n_ctx() == effective_n_ctx:
                return current_model
            else:
                logger.info(f"GGUF: Reloading {model_alias} with new n_ctx={effective_n_ctx}")
                self.unload(model_alias)

        # Hardware Check
        n_gpu_layers = meta.get("num_layers", -1) if self.backend == "gpu" else 0
        
        params_b = meta.get("num_params")
        quant = meta.get("quantization")
        if params_b and quant:
            from ..routers.calculator import estimate_vram_ram
            from ..resource_manager import resource_manager
            from fastapi import HTTPException
            
            estimates = estimate_vram_ram(params_b, quant, effective_n_ctx, n_gpu_layers)
            req_vram_mb = estimates["estimated_vram_gb"] * 1024
            req_ram_mb = estimates["estimated_ram_gb"] * 1024
            
            avail_vram_mb = resource_manager.get_available_vram_mb()
            
            # Verificamos apenas se cabe na VRAM caso o backend seja GPU
            if self.backend == "gpu" and req_vram_mb > avail_vram_mb:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Model does not fit on your GPU. Required VRAM: {req_vram_mb:.2f}MB, Available VRAM: {avail_vram_mb:.2f}MB."
                )

        try:
            logger.info(f"GGUF: Loading model {model_alias} on {self.backend} (n_ctx={effective_n_ctx})")
            
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
                    raise ValueError(f"GGUF: Vision model {model_alias} is missing mmproj metadata.")
                
                mmproj_path = self._ensure_downloaded(mmproj_id, mmproj_filename)
                
                vision_type = detect_model_type(meta["model_id"] + meta["filename"])
                logger.info(f"GGUF: Vision enabled. Detected type: {vision_type} with mmproj {mmproj_path}")
                
                if vision_type == "moondream":
                    from llama_cpp.llama_chat_format import MoondreamChatHandler
                    params["chat_handler"] = MoondreamChatHandler(clip_model_path=mmproj_path)
                elif vision_type == "nanollava":
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                    params["chat_handler"] = NanoLlavaChatHandler(clip_model_path=mmproj_path)
                else: # fallback to llava or llava-v1.6
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)

            model = Llama(**params)
            self.active_models[model_alias] = model
            return model
        except Exception as e:
            logger.error(f"GGUF: Failed to load {model_alias}: {e}")
            raise e

    def _build_raw_prompt(self, messages: List[Dict[str, Any]], template_name: str, force_reasoning: bool = False, custom_template: str = None):
        prompt = ""
        stop_words = []
        template_name = str(template_name).lower() if template_name else ""

        # 1. Aplica o Custom Template se existir
        if custom_template:
            try:
                import jinja2
                # Renderiza usando Jinja2 (padrão do HuggingFace/Ollama)
                template = jinja2.Template(custom_template)
                prompt = template.render(messages=messages)
                
                # Se o template não deixar o turno aberto, forçamos a abertura básica
                if not prompt.strip().endswith("<think>") and force_reasoning:
                    pass # O prefill no passo 3 vai adicionar o think adequadamente
                    
            except ImportError:
                logger.warning("GGUF: jinja2 não instalado. Ignorando custom_template e caindo no fallback.")
            except Exception as e:
                logger.error(f"GGUF: Erro ao renderizar custom_template: {e}")

        # 2. Se não houver custom_template (ou falhou), usa as lógicas nativas
        if not prompt:
            if "llama-3" in template_name:
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                stop_words = ["<|eot_id|>"]

            elif "chatml" in template_name or "qwen" in template_name:
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
                prompt += "<|im_start|>assistant\n"
                stop_words = ["<|im_end|>"]

            else:
                # Fallback genérico
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    prompt += f"{role.capitalize()}: {content}\n"
                
                prompt += "Assistant: "
                stop_words = ["User:", "Assistant:"]

        # 3. Aplica o Prefill de Raciocínio (se solicitado)
        if force_reasoning:
            # Evita adicionar quebras de linha duplicadas
            if prompt.endswith("\n\n"):
                prompt += "<think>\n"
            elif prompt.endswith("\n"):
                prompt += "<think>\n"
            else:
                prompt += "\n<think>\n"

        return prompt, stop_words

    def run_chat(self, model_name: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        headers = kwargs.pop("headers", {})
        header_n_ctx_str = headers.get("x-context-window")
        
        n_ctx = kwargs.pop("num_ctx", None)
        
        if header_n_ctx_str:
            try:
                n_ctx = int(header_n_ctx_str)
            except ValueError:
                logger.warning(f"GGUF: Invalid x-context-window header value: {header_n_ctx_str}")

        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")
            
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        meta = self._get_model_metadata(model_name)
        model = self.load(model_alias=model_name, n_ctx=n_ctx)

        # Extração de Configurações
        force_reasoning = meta.get("force_reasoning", False)
        custom_template = meta.get("custom_template")
        template_name = meta.get("template", "")

        logger.info(f"GGUF: Executando Unified Completion Adapter para {model_name} (force_reasoning={force_reasoning}, custom_template={bool(custom_template)})")
        
        # 1. Monta o prompt cru e as stop words
        raw_prompt, template_stops = self._build_raw_prompt(
            messages=messages, 
            template_name=template_name,
            force_reasoning=force_reasoning,
            custom_template=custom_template
        )
        
        # 2. Mescla stop words (Agent + Template)
        stops = kwargs.get("stop", [])
        if isinstance(stops, str):
            stops = [stops]
            
        stops.extend(template_stops)
        kwargs["stop"] = list(set(stops)) # Remove duplicatas

        try:
            import time
            import uuid
            
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = -1
            
            # 3. CHAMA A API BRUTA PARA TODOS OS MODELOS
            response = model.create_completion(
                prompt=raw_prompt,
                stream=stream,
                **kwargs
            )

            if stream:
                prompt_tokens = len(model.tokenize(raw_prompt.encode("utf-8")))
                base_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

                def stream_adapter():
                    full_text = ""
                    
                    yield {
                        "id": base_id,
                        "model": model_name,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": "<think>\n"},
                            "finish_reason": None
                        }]
                    }
                    
                    for chunk in response:
                        if not chunk.get("choices"):
                            yield chunk
                            continue
                        
                        text = chunk["choices"][0].get("text", "")
                        full_text += text
                        finish_reason = chunk["choices"][0].get("finish_reason")
                        
                        out_chunk = {
                            "id": chunk.get("id", base_id),
                            "model": chunk.get("model", model_name),
                            "object": "chat.completion.chunk",
                            "created": chunk.get("created", int(time.time())),
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": finish_reason
                            }]
                        }
                        
                        yield out_chunk
                        
                        # 4. Padrão OpenAI: Finaliza e injeta o usage matemático perfeito
                        if finish_reason is not None:
                            comp_tokens = len(model.tokenize(full_text.encode('utf-8')))
                            yield {
                                "id": out_chunk["id"],
                                "model": out_chunk["model"],
                                "object": "chat.completion.chunk",
                                "created": out_chunk["created"],
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": comp_tokens,
                                    "total_tokens": prompt_tokens + comp_tokens
                                }
                            }
                return stream_adapter()
            
            else:
                # 5. Saída Não-Stream
                text = response["choices"][0].get("text", "")
                prompt_tokens = len(model.tokenize(raw_prompt.encode("utf-8")))
                comp_tokens = len(model.tokenize(text.encode("utf-8")))
                
                return {
                    "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:10]}"),
                    "model": response.get("model", model_name),
                    "object": "chat.completion",
                    "created": response.get("created", int(time.time())),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": response["choices"][0].get("finish_reason", "stop")
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "total_tokens": prompt_tokens + comp_tokens
                    }
                }

        except Exception as e:
            logger.error(f"GGUF: Unified Completion Adapter failed: {e}")
            raise e
    
    def unload(self, model_name: str):
        if model_name in self.active_models:
            del self.active_models[model_name]
            import gc
            gc.collect()
            logger.info(f"GGUF: Unloaded {model_name}")
