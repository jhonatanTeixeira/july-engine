import os
import json
import logging
from typing import Any, Dict, List, Optional
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
        
        # Ensures model is downloaded
        model_path = self._ensure_downloaded(meta["model_id"], meta["filename"])

        if model_alias in self.active_models:
            current_model = self.active_models[model_alias]
            if n_ctx is None or current_model.n_ctx() == n_ctx:
                return current_model
            else:
                logger.info(f"GGUF: Reloading {model_alias} with new n_ctx={n_ctx}")
                self.unload(model_alias)

        try:
            logger.info(f"GGUF: Loading model {model_alias} on {self.backend} (n_ctx={n_ctx})")
            n_gpu_layers = -1 if self.backend == "gpu" else 0
            
            params = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx or int(os.environ.get("LLM_CTX_TOKENS", 2048)),
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

    def run_chat(self, model_name: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        # Interpret model_name as alias now
        n_ctx = kwargs.pop("num_ctx", None)
        
        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")
            
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        model = self.load(model_alias=model_name, n_ctx=n_ctx)
        
        try:
            return model.create_chat_completion(
                messages=messages,
                stream=stream,
                **kwargs
            )
        except Exception as e:
            logger.error(f"GGUF: Chat execution failed: {e}")
            raise e

    def unload(self, model_name: str):
        if model_name in self.active_models:
            del self.active_models[model_name]
            import gc
            gc.collect()
            logger.info(f"GGUF: Unloaded {model_name}")
