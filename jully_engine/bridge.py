import asyncio
import logging
import uuid
import time
import json
from fastapi import HTTPException
from typing import Any, Dict, Optional, Union, AsyncGenerator, List
from .orchestrators.gpu_orchestrator import gpu_orchestrator
from .orchestrators.cpu_orchestrator import cpu_orchestrator
from .orchestrators.api_orchestrator import api_orchestrator

logger = logging.getLogger("JulyEngine.Bridge")

class Bridge:
    """
    Consolidated Bridge that routes requests to the appropriate orchestrator
    and normalizes OpenAI/Anthropic formats.
    """
    def __init__(self):
        self.orchestrators = {
            "gpu": gpu_orchestrator,
            "cpu": cpu_orchestrator,
            "api": api_orchestrator
        }

    async def start(self):
        for name, orch in self.orchestrators.items():
            await orch.start()

    async def stop(self):
        for name, orch in self.orchestrators.items():
            await orch.stop()

    def get_orchestrator(self, headers: Dict[str, str]):
        backend = headers.get("x-backend")
        if not backend:
            raise HTTPException(status_code=400, detail="Missing x-backend header")
            
        backend = backend.lower()
        if backend not in self.orchestrators:
            raise HTTPException(status_code=400, detail=f"Unknown backend {backend}")
            
        return self.orchestrators[backend]

    async def process_openai_chat(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        
        messages = payload.get("messages", [])
        stream = payload.get("stream", False)
        model_name = payload.get("model", "default")
        
        is_vision = False
        last_message = messages[-1] if messages else None
        if last_message and isinstance(last_message.get("content"), list):
            if any(p.get("type") == "image_url" for p in last_message["content"]):
                is_vision = True

        task_type = "vision_chat" if is_vision else "text_chat"
        
        future = orch.submit_task(task_type, payload)
        response = await asyncio.wrap_future(future)

        if not stream:
            if isinstance(response, str):
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
            return response

        async def openai_generator():
            for chunk in response:
                chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk)
                yield f"data: {json.dumps(chunk_dict)}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return openai_generator()

    async def process_anthropic_message(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        
        messages = payload.get("messages", [])
        stream = payload.get("stream", False)
        
        is_vision = False
        last_message = messages[-1] if messages else None
        if last_message and isinstance(last_message.get("content"), list):
            if any(p.get("type") == "image" for p in last_message["content"]):
                is_vision = True
        
        task_type = "vision_chat" if is_vision else "text_chat"
        
        future = orch.submit_task(task_type, payload)
        response = await asyncio.wrap_future(future)
        
        if not stream:
            if isinstance(response, str):
                return {
                    "id": f"msg_{uuid.uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "model": payload.get("model", "claude-3"),
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            return response

        async def anthropic_generator():
            msg_id = f"msg_{uuid.uuid4()}"
            model_name = payload.get("model", "claude-3")
            
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            
            for chunk in response:
                chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk)
                if 'choices' in chunk_dict:
                    delta = chunk_dict['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                
                await asyncio.sleep(0)
            
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return anthropic_generator()

    async def process_embeddings(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[List[float]]:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        
        future = orch.submit_task("embedding", payload)
        response = await asyncio.wrap_future(future)
        
        if isinstance(response, list) and len(response) > 0 and not isinstance(response[0], list):
            return [response]
        return response

    async def process_tts(self, payload: Dict[str, Any], headers: Dict[str, str]) -> bytes:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        future = orch.submit_task("tts", payload)
        return await asyncio.wrap_future(future)

    async def process_stt(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        future = orch.submit_task("stt", payload)
        return await asyncio.wrap_future(future)

    async def process_image_edit(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        future = orch.submit_task("pix2pix", payload)
        return await asyncio.wrap_future(future)

    async def process_image_generation(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        future = orch.submit_task("image_generation", payload)
        return await asyncio.wrap_future(future)

bridge = Bridge()
