from __future__ import annotations
import logging
import threading
from typing import Callable, Dict, Any, Optional, Type, Tuple
from .models.base_model import BaseModel
from .adapters.adapter_base import AdapterBase



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

def _get_entity_adapter():
    from .adapters.entity_adapter import EntityAdapter
    return EntityAdapter


_ADAPTER_REGISTRY: Dict[str, Callable[[], Type[AdapterBase]]] = {
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
    "video_generation": _get_image_adapter,
    "video_description": _get_vision_adapter,
    "entity_extraction": _get_entity_adapter,
}


class ModelLoader:
    def __init__(self):
        self.instances: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _resolve_adapter_cls(self, task_type: str) -> Tuple[str, Type[AdapterBase]]:
        adapter_getter = _ADAPTER_REGISTRY.get(task_type)

        if not adapter_getter:
            raise ValueError(f"ModelLoader: unknown task_type '{task_type}'")

        adapter_cls = adapter_getter()

        return adapter_cls.get_engine_type(task_type), adapter_cls
    

    def get(self, task_type: str, backend: str | None = None, model_tag: str | None = None) -> AdapterBase:
        """
        Returns a cached model instance for (task_type, backend, model_tag).
        The instance is created using the resolved adapter class.
        """

        with self.lock:
            from .services.models_service import model_service

            engine, adapter_cls = self._resolve_adapter_cls(task_type)
            model_meta: Optional[dict] = None
            engine_settings = model_service.get_setting(engine)

            if not model_tag or model_tag == "default":
                setting = engine_settings

                if setting:
                    model_meta = setting if isinstance(setting, dict) else next((p for p in setting if p.get("is_default", False)), setting[0])
                    model_tag = model_meta.get("alias") or model_meta.get("model")
            else:
                if isinstance(engine_settings, list):
                    default_setting = next((p for p in engine_settings if p.get("is_default", False)), engine_settings[0])
                    model_meta = next(
                        (p for p in engine_settings if p.get("alias") == model_tag or p.get("model") == model_tag),
                        default_setting,
                    )
                    # Normalize model_tag to the actual resolved model to prevent cache key mismatches
                    # where e.g. "multilingual_e5" maps to the default chat model (Qwen) under a wrong key.
                    model_tag = model_meta.get("alias") or model_meta.get("model") or model_tag
                elif engine_settings and engine_settings.get("model") == model_tag:
                    model_meta = engine_settings
                elif engine_settings and engine_settings.get("alias") == model_tag:
                    model_meta = engine_settings
                else:
                    model_meta = {"model": model_tag, "backend": backend}

            if not model_meta:
                raise ValueError(f'no model or setting could be found for {task_type}')

            if backend:
                model_meta["backend"] = backend

            # Use the resolved backend for the cache key so that requests with/without
            # explicit x-backend headers always hit the same cached instance.
            resolved_backend = backend or model_meta.get("backend") or "gpu"
            key = f"{task_type}_{resolved_backend}_{model_tag}"

            if key in self.instances:
                return self.instances[key]

            instance = adapter_cls(task_type, backend=resolved_backend, model_meta=model_meta)

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
