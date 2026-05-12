from __future__ import annotations
import logging
import threading
from typing import Callable, Dict, Any, Optional, Type

logger = logging.getLogger("JulyEngine.ModelLoader")

# ---------------------------------------------------------------------------
# Lazy registry for Adapter Classes
# ---------------------------------------------------------------------------

def _get_chat_adapter():
    from .adapters.chat_adapter import ChatAdapter
    return ChatAdapter

def _get_rag_adapter():
    from .adapters.rag_adapter import RagAdapter
    return RagAdapter

def _get_tts_adapter():
    from .adapters.tts_adapter import TTSAdapter
    return TTSAdapter

def _get_stt_adapter():
    from .adapters.stt_adapter import STTAdapter
    return STTAdapter

def _get_vision_adapter():
    from .adapters.vision_adapter import VisionAdapter
    return VisionAdapter

def _get_image_adapter():
    from .adapters.image_adapter import ImageAdapter
    return ImageAdapter

def _get_search_adapter():
    from .adapters.search_adapter import SearchAdapter
    return SearchAdapter


_ADAPTER_REGISTRY: Dict[str, Callable[[], Type]] = {
    "text_chat": _get_chat_adapter,
    "vision_chat": _get_vision_adapter,
    "tts": _get_tts_adapter,
    "stt": _get_stt_adapter,
    "embeddings": _get_rag_adapter,
    "rag_add": _get_rag_adapter,
    "rag_batch_add": _get_rag_adapter,
    "rag_vector_add": _get_rag_adapter,
    "rag_search": _get_rag_adapter,
    "rag_update": _get_rag_adapter,
    "rag_delete": _get_rag_adapter,
    "rag_list": _get_rag_adapter,
    "rag_smart_search": _get_rag_adapter,
    "image_generation": _get_image_adapter,
    "image_edit": _get_image_adapter,
    "image_resize": _get_image_adapter,
    "image_remove_background": _get_image_adapter,
    "web_search": _get_search_adapter,
    "code_search": _get_search_adapter,
    "video_description": _get_vision_adapter
}


class ModelLoader:
    def __init__(self):
        self.instances: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def get(self, task_type: str, backend: str, model_meta: dict):
        """
        Returns a cached model instance for (task_type, backend, model_tag).
        The instance is created using the resolved adapter class.
        """
        model_tag = model_meta.get("alias") or model_meta.get("model")
        key = f"{task_type}_{backend}_{model_tag}"

        with self.lock:
            if key in self.instances:
                return self.instances[key]
            
            # Resolve a classe do adapter
            registry_func = _ADAPTER_REGISTRY.get(task_type)
            if not registry_func:
                raise ValueError(f"ModelLoader: unknown task_type '{task_type}'")
            
            adapter_cls = registry_func()
            
            # Instanciação cirúrgica baseada na assinatura do adapter
            # (Alguns pedem task_type no construtor, outros não)
            if adapter_cls.__name__ in ("RagAdapter", "VisionAdapter", "ImageAdapter", "SearchAdapter"):
                instance = adapter_cls(task_type=task_type, backend=backend, model_meta=model_meta)
            else:
                instance = adapter_cls(backend=backend, model_meta=model_meta)
                
            self.instances[key] = instance
            logger.info(f"[ModelLoader] model={model_tag} backend={backend} task={task_type}")

            return instance

    def delete_instance(self, backend: str, model_alias: str):
        """Remove todas as instâncias que casam com o backend e alias fornecidos."""
        suffix = f"_{backend}_{model_alias}"
        
        with self.lock:
            keys_to_del = [k for k in self.instances.keys() if k.endswith(suffix)]
            for k in keys_to_del:
                logger.info(f"[ModelLoader] Deleting instance cache key: {k}")
                self.instances.pop(k, None)


model_loader = ModelLoader()
