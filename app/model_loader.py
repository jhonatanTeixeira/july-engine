from __future__ import annotations
import logging
import threading
from typing import Callable, Dict, Any

logger = logging.getLogger("JulyEngine.ModelLoader")

# ---------------------------------------------------------------------------
# Lazy registry
#
# Values are zero-arg factory functions that import and return the class only
# when first needed.  Nothing heavy is imported at module load time, so
# environments that lack optional libs (torch, TTS, kokoro, …) won't break
# during API startup.
#
# To register a new engine:
#   1. Implement BaseModel in app/models/ or app/adapters/
#   2. Add a factory lambda here — nothing else needs to change.
# ---------------------------------------------------------------------------

def _chat_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.chat_adapter import ChatAdapter

    return ChatAdapter(backend, model_meta)

def _rag_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.rag_adapter import RagAdapter

    return RagAdapter(task_type, backend, model_meta)

def _tts_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.tts_adapter import TTSAdapter
    return TTSAdapter(backend, model_meta)

def _stt_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.stt_adapter import STTAdapter
    return STTAdapter(backend, model_meta)

def _vision_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.vision_adapter import VisionAdapter
    return VisionAdapter(task_type, backend, model_meta)

def _image_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.image_adapter import ImageAdapter
    return ImageAdapter(task_type, backend, model_meta)

def _search_adapter(task_type: str, backend: str, model_meta: dict):
    from .adapters.search_adapter import SearchAdapter
    return SearchAdapter(task_type, backend, model_meta)


_ENGINE_FACTORIES: Dict[str, Callable] = {
    "text_chat": _chat_adapter,
    "vision_chat": _vision_adapter,
    "tts": _tts_adapter,
    "stt": _stt_adapter,
    "embeddings": _rag_adapter,
    "rag_add": _rag_adapter,
    "rag_batch_add": _rag_adapter,
    "rag_vector_add": _rag_adapter,
    "rag_search": _rag_adapter,
    "rag_update": _rag_adapter,
    "rag_delete": _rag_adapter,
    "rag_list": _rag_adapter,
    "rag_smart_search": _rag_adapter,
    "image_generation": _image_adapter,
    "image_edit": _image_adapter,
    "image_resize": _image_adapter,
    "image_remove_background": _image_adapter,
    "web_search": _search_adapter,
    "code_search": _search_adapter,
    "video_description": _vision_adapter
}


class ModelLoader:
    def __init__(self):
        self.instances: Dict[str, Any] = {}
        self._class_cache: Dict[str, type] = {}
        self.lock = threading.Lock()

    def get(self, task_type: str, backend: str, model_meta: dict):
        """
        Returns a cached model instance for (backend, model_tag).
        The class is resolved from model_meta["engine"] via lazy factory;
        the instance is cached so the same model isn't loaded twice.
        """
        model_tag = model_meta.get("alias") or model_meta.get("model")
        key = f"{backend}_{model_tag}"

        with self.lock:
            if key in self.instances:
                return self.instances[key]
            
            factory = _ENGINE_FACTORIES.get(task_type)
            instance = factory(task_type, backend, model_meta)
            self.instances[key] = instance
            logger.info(f"[ModelLoader] model={model_tag} backend={backend}")

            return instance

    def delete_instance(self, backend: str, model_alias: str):
        key = f"{backend}_{model_alias}"
        with self.lock:
            self.instances.pop(key, None)


model_loader = ModelLoader()
