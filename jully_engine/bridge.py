import asyncio
import logging
import uuid
import time
import json
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
        backend = headers.get("x-backend", "gpu").lower()
        if backend not in self.orchestrators:
            logger.warning(f"Bridge: Unknown backend {backend}, falling back to gpu")
            backend = "gpu"
        return self.orchestrators[backend]

    async def process_openai_chat(self, request_data: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        headers = request_data.get("headers", {})
        messages = request_data.get("messages", [])
        model_name = request_data.get("model", "default")
        stream = request_data.get("stream", False)
        
        # Check for multimodal (vision)
        last_message = messages[-1] if messages else None
        is_vision = False
        payload = request_data

        if last_message and isinstance(last_message.get("content"), list):
            content_parts = last_message["content"]
            image_part = next((p for p in content_parts if p.get("type") == "image_url"), None)
            text_part = next((p for p in content_parts if p.get("type") == "text"), None)

            if image_part:
                is_vision = True
                image_url = image_part["image_url"]["url"]
                # If it's data URI, extract base64 part
                if image_url.startswith("data:"):
                    try:
                        image_data = image_url.split(",")[1]
                    except IndexError:
                        image_data = image_url
                else:
                    image_data = image_url

                payload = {
                    "prompt": text_part.get("text", "") if text_part else "describe this image", 
                    "image": image_data,
                    "model": model_name,
                    "headers": headers
                }

        task_type = "vision_chat" if is_vision else "text_chat"
        orch = self.get_orchestrator(headers)
        backend = headers.get("x-backend", "gpu").lower()

        # Final payload decision
        final_payload = payload
        if backend == "api":
            # litellm expects the full OpenAI structure
            final_payload = request_data
            final_payload['headers'] = headers

        future = orch.submit_task(task_type, final_payload)
        response = await asyncio.wrap_future(future)

        if not stream:
            # Normalize non-streaming response if it's just a string from local models
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

        # Streaming path
        async def openai_generator():
            # If response is already a generator (from litellm or llama-cpp)
            for chunk in response:
                chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk)
                yield f"data: {json.dumps(chunk_dict)}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return openai_generator()

    async def process_anthropic_message(self, request_data: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        headers = request_data.get("headers", {})
        messages = request_data.get("messages", [])
        stream = request_data.get("stream", False)
        
        # Detect if it's vision
        last_message = messages[-1] if messages else None
        is_vision = False
        payload = request_data

        if last_message and isinstance(last_message.get("content"), list):
            content_parts = last_message["content"]
            image_part = next((p for p in content_parts if p.get("type") == "image"), None)
            text_part = next((p for p in content_parts if p.get("type") == "text"), None)

            if image_part:
                is_vision = True
                image_source = image_part.get("source", {}).get("data", "")
                payload = {
                    "prompt": text_part.get("text", "") if text_part else "describe this image", 
                    "image": image_source,
                    "model": request_data.get("model"),
                    "headers": headers
                }
        
        task_type = "vision_chat" if is_vision else "text_chat"
        orch = self.get_orchestrator(headers)
        future = orch.submit_task(task_type, payload)
        response = await asyncio.wrap_future(future)
        
        if not stream:
            if isinstance(response, str):
                return {
                    "id": f"msg_{uuid.uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "model": request_data.get("model", "claude-3"),
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            return response

        # Streaming path
        async def anthropic_generator():
            msg_id = f"msg_{uuid.uuid4()}"
            model_name = request_data.get("model", "claude-3")
            
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            
            for chunk in response:
                chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk)
                # litellm chunks are OpenAI format, need conversion to Anthropic for streaming
                if 'choices' in chunk_dict:
                    delta = chunk_dict['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                
                await asyncio.sleep(0)
            
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return anthropic_generator()

    async def process_embeddings(self, inputs: Union[str, List[str]], model: str, headers: Dict[str, str]) -> List[List[float]]:
        orch = self.get_orchestrator(headers)
        future = orch.submit_task("embedding", {"input": inputs, "model": model, "headers": headers})
        response = await asyncio.wrap_future(future)
        
        # Normalize local response (often a single list for one input) to list of lists
        if isinstance(response, list) and len(response) > 0 and not isinstance(response[0], list):
            return [response]
        return response

    async def process_tts(self, text: str, voice: str, model: str, headers: Dict[str, str]) -> bytes:
        orch = self.get_orchestrator(headers)
        future = orch.submit_task("tts", {"text": text, "voice": voice, "model": model, "headers": headers})
        return await asyncio.wrap_future(future)

    async def process_stt(self, audio_bytes: bytes, model: str, language: Optional[str], headers: Dict[str, str]) -> str:
        orch = self.get_orchestrator(headers)
        future = orch.submit_task("stt", {"audio": audio_bytes, "model": model, "language": language, "headers": headers})
        return await asyncio.wrap_future(future)

    async def process_image_edit(self, image_data: str, prompt: str, model: Optional[str], headers: Dict[str, str]) -> str:
        orch = self.get_orchestrator(headers)
        future = orch.submit_task("pix2pix", {"prompt": prompt, "image": image_data, "model": model, "headers": headers})
        return await asyncio.wrap_future(future)

    async def process_image_generation(self, prompt: str, model: Optional[str], headers: Dict[str, str]) -> str:
        orch = self.get_orchestrator(headers)
        # For local models, image generation might fall back to pix2pix with a dummy image 
        # or use a dedicated stable diffusion strategy.
        future = orch.submit_task("image_generation", {"prompt": prompt, "model": model, "headers": headers})
        return await asyncio.wrap_future(future)

bridge = Bridge()
