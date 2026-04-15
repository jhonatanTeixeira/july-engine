import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional

from fastapi import HTTPException
from .models_service import ModelsService
from ..orchestrators.api_orchestrator import api_orchestrator


class OrchestratorContainer:
    def __init__(self):
        self.model_service = ModelsService()
        
        self.orchestrators = {
            "api": api_orchestrator
        }
        
        if os.environ.get("DISABLE_GPU", "false").lower() != "true":
            from ..orchestrators.gpu_orchestrator import gpu_orchestrator
            self.orchestrators["gpu"] = gpu_orchestrator
        else:
            self.orchestrators["gpu"] = None
            
        if os.environ.get("DISABLE_CPU", "false").lower() != "true":
            from ..orchestrators.cpu_orchestrator import cpu_orchestrator
            self.orchestrators["cpu"] = cpu_orchestrator
        else:
            self.orchestrators["cpu"] = None
            
    def resolve_backend(self, task_type: str, payload: Dict):
        if task_type == 'text_chat':
            config = self.model_service.resolve_by_settings(payload.get("model"))
            
            if not config:
                raise HTTPException(422, detail=f"Model {payload.get('model')} is not configured on presets")
        else:
            config = self.model_service.backend.get_setting(task_type) or {}
        
        headers = payload.setdefault("headers", {})
        backend = headers.get("x-backend", config.get("backend"))
        
        if backend == "image_edit_model":
            payload["is_image_edit_route"] = True
            edit_config = self.model_service.backend.get_setting("IMAGE_EDIT")

            if edit_config:
                backend = edit_config.get("backend", "api")
                if not payload.get("model"):
                    payload["model"] = edit_config.get("model")

            else:
                backend = "api"
        
        if not backend:
            raise HTTPException(status_code=400, detail="Missing x-backend header or model not configured")

        if not payload.get("model", None):
            payload["model"] = config.get("model")
            
        return backend.lower(), payload["model"]

    def get_orchestrator(self, task_type: str, payload: Dict[str, str]):
        backend, _ = self.resolve_backend(task_type, payload)
        
        if backend not in self.orchestrators:
            raise HTTPException(status_code=400, detail=f"Unknown backend {backend}")
            
        orch = self.orchestrators.get(backend, None)
        
        if orch is None:
            raise HTTPException(status_code=400, detail=f"Backend {backend} is disabled on this engine")
            
        return orch


@dataclass
class InferenceHelper:
    orchestrator_container: OrchestratorContainer
    
    def __post_init__(self):
        self.task_mapping = {
            "text_chat": "text_chat",
            "tts": "TTS",
            "stt": "STT",
            "vision_chat": "VISION",
            "embeddings": "EMBEDDINGS",
            "pix2pix": "IMAGE_EDIT",
            "image_resize": "RESIZE",
            "image_generation": "IMAGE_CREATE",
            "search_web": "WEB_SEARCH",
            "search_code": "REPOSITORY_SEARCH",
            # RAG / Memory Tasks
            "rag_add": "EMBEDDINGS",
            "rag_batch_add": "EMBEDDINGS",
            "rag_search": "EMBEDDINGS",
            "rag_vector_add": "EMBEDDINGS",
            "rag_search_details": "EMBEDDINGS",
            "rag_update": "EMBEDDINGS",
            "rag_delete": "EMBEDDINGS",
            "rag_list": "EMBEDDINGS",
            "rag_smart_search": "EMBEDDINGS"
        }
    
    async def _await_orch_task(self, future_or_coro):
        if asyncio.iscoroutine(future_or_coro):
            res = await future_or_coro
        else:
            res = await asyncio.wrap_future(future_or_coro)
        
        if asyncio.iscoroutine(res):
            return await res

        return res

    async def process(self, task_type: str, payload: Dict):
        orch = self.orchestrator_container.get_orchestrator(self.task_mapping[task_type], payload)
        
        return await self._await_orch_task(orch.submit_task(task_type, payload))


inference_helper = InferenceHelper(OrchestratorContainer())


@dataclass
class MultiModalHelper:
    payload: Dict
    
    def __post_init__(self):
        self.messages = self.payload.get("messages")
        
    def _get_multimodal_content(self, messages: Optional[List[Dict]] = None):
        target_messages = messages if messages is not None else self.messages
        
        for message in target_messages:
            content = message.get("content", "")
            
            if isinstance(content, list):
                for part in content:
                    yield part
                    
    def get_images(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return [
            part for part in self._get_multimodal_content(messages) 
            if part.get("type", "text") == "image_url"
        ]
        
    def get_last_image(self) -> Optional[Dict]:
        images = self.get_images()
        return images[-1] if images else None
        
    def get_audios(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return [
            part for part in self._get_multimodal_content(messages) 
            if part.get("type", "text") in ["audio_url", "input_audio"]
        ]
        
    def get_last_audio(self) -> Optional[Dict]:
        audios = self.get_audios()
        return audios[-1] if audios else None

    def _filter_content_by_type(self, types_to_remove: List[str], messages: Optional[List[Dict]] = None) -> List[Dict]:
        msgs = deepcopy(messages if messages is not None else self.messages)
        
        for msg in msgs:
            content = msg.get("content", "")
            
            if isinstance(content, list):
                msg["content"] = [part for part in content if part.get("type") not in types_to_remove]
        
        return msgs

    def filter_images(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["image_url"], messages)
        
    def filter_audios(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["audio_url", "input_audio"], messages)
        
    def last_message_images(self) -> List[Dict]:
        return self.get_images([self.messages[-1]]) if self.messages else []
        
    def last_message_audios(self) -> List[Dict]:
        return self.get_audios([self.messages[-1]]) if self.messages else []

    async def _await_orch_task(self, future_or_coro):
        if asyncio.iscoroutine(future_or_coro):
            res = await future_or_coro
        else:
            res = await asyncio.wrap_future(future_or_coro)
        
        if asyncio.iscoroutine(res):
            return await res

        return res

    async def process_vision(self):
        if (last_images := self.last_message_images()):
            vision_content = deepcopy(last_images) 
            vision_content.append({
                "type": "text", 
                "text": "Describe this image in detail."
            })

            vision_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": vision_content
                    }
                ],
                "headers": {}
            }

            response: List[str] = await inference_helper.process('vision_chat', vision_payload)
            last_content: list = self.messages[-1].setdefault("content", [])
            
            format_content = lambda content: {"type": "text", "text": f"[User sent an image]: {content}"}
            
            for content in response:
                last_content.append(format_content(content))
    
    async def process_transcription(self) -> None:
        if (last_audios := self.last_message_audios()):
            audio_obj = last_audios[0]

            if audio_obj.get("type") == "input_audio":
                b64_str = audio_obj.get("input_audio", {}).get("data", "")
            else:
                b64_str = audio_obj.get("audio_url", {}).get("url", "").split(",")[-1]

            audio_bytes = base64.b64decode(b64_str)

            transcription_payload = {
                "audio": audio_bytes,
                "headers": {}
            }

            response: str = await inference_helper.process('stt', transcription_payload)
            last_content: list = self.messages[-1].setdefault("content", [])
            
            last_content.append({"type": "text", "text": f"[User sent audio]: {response}"})