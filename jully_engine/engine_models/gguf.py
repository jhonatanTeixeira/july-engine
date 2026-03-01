import os
import logging
from typing import Any, Dict, List, Optional
from llama_cpp import Llama

logger = logging.getLogger("JulyEngine.Models.GGUF")

class GGUF:
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.active_models = {}

    def _get_model_path(self, model_name: str) -> str:
        if not model_name.endswith(".gguf"):
            model_name += ".gguf"
        return os.path.join(self.models_dir, model_name)

    def load(self, model_name: str, is_vision: bool = False, n_ctx: Optional[int] = None):
        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found at {model_path}")

        # If model is loaded but with different n_ctx, we might need to reload or just return
        # For simplicity, if it's already loaded we return it. 
        # But if n_ctx is provided and different, we should probably reload.
        if model_name in self.active_models:
            current_model = self.active_models[model_name]
            # Simple check if reload needed (optional optimization)
            if n_ctx is None or current_model.n_ctx() == n_ctx:
                return current_model
            else:
                logger.info(f"GGUF: Reloading {model_name} with new n_ctx={n_ctx}")
                self.unload(model_name)

        try:
            logger.info(f"GGUF: Loading model {model_name} on {self.backend} (n_ctx={n_ctx})")
            n_gpu_layers = -1 if self.backend == "gpu" else 0
            
            params = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx or int(os.environ.get("LLM_CTX_TOKENS", 2048)),
                "verbose": False
            }

            if is_vision:
                mmproj_path = model_path.replace(".gguf", "-mmproj.gguf")
                if os.path.exists(mmproj_path):
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path)
                    logger.info(f"GGUF: Vision enabled with mmproj {mmproj_path}")

            model = Llama(**params)
            self.active_models[model_name] = model
            return model
        except Exception as e:
            logger.error(f"GGUF: Failed to load {model_name}: {e}")
            raise e

    def run_chat(self, model_name: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        is_vision = any(isinstance(m.get('content'), list) for m in messages)
        
        # Extract num_ctx from kwargs (passed from extra_body)
        n_ctx = kwargs.pop("num_ctx", None)
        
        # Map OpenAI-style repetition_penalty to llama-cpp repeat_penalty
        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")
            
        # Filter out None values to avoid ctypes errors in llama-cpp
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        model = self.load(model_name, is_vision=is_vision, n_ctx=n_ctx)
        
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
