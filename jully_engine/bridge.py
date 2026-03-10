import asyncio
import logging
import uuid
import time
import json
import os
from fastapi import HTTPException
from typing import Any, Dict, Optional, Union, AsyncGenerator, List
from .orchestrators.api_orchestrator import api_orchestrator

logger = logging.getLogger("JulyEngine.Bridge")

class Bridge:
    """
    Consolidated Bridge that routes requests to the appropriate orchestrator
    and normalizes OpenAI/Anthropic formats.
    """
    def __init__(self):
        self.orchestrators = {
            "api": api_orchestrator
        }
        
        if os.environ.get("DISABLE_GPU", "false").lower() != "true":
            from .orchestrators.gpu_orchestrator import gpu_orchestrator
            self.orchestrators["gpu"] = gpu_orchestrator
        else:
            self.orchestrators["gpu"] = None
            
        if os.environ.get("DISABLE_CPU", "false").lower() != "true":
            from .orchestrators.cpu_orchestrator import cpu_orchestrator
            self.orchestrators["cpu"] = cpu_orchestrator
        else:
            self.orchestrators["cpu"] = None

    def _enrich_headers_and_payload(self, task_key: str, payload: Dict[str, Any], headers: Dict[str, str]):
        try:
            from .persistence import get_backend
            backend_db = get_backend()
            
            config = None
            if task_key in ["text_chat", "vision_chat", "embedding"]:
                preset_alias = payload.get("model")
                text_presets = backend_db.get_setting("TEXT_PRESETS") or []
                config = next((p for p in text_presets if p.get("alias") == preset_alias), None)
                if not config and text_presets:
                    config = text_presets[0]
            else:
                mapping = {
                    "tts": "TTS",
                    "stt": "STT",
                    "pix2pix": "IMAGE_EDIT",
                    "image_generation": "IMAGE_CREATE",
                    "search_web": "WEB_SEARCH",
                    "search_code": "REPOSITORY_SEARCH"
                }
                setting_key = mapping.get(task_key)
                if setting_key:
                    config = backend_db.get_setting(setting_key)

            if config:
                if "x-backend" not in headers and "backend" in config:
                    headers["x-backend"] = config["backend"]
                if "x-base-url" not in headers and "base_url" in config:
                    headers["x-base-url"] = config["base_url"]
                
                # Check for standard Authorization or x-api-key
                has_auth = "authorization" in headers or "x-api-key" in headers
                if not has_auth and "api_key" in config and config["api_key"]:
                    headers["x-api-key"] = config["api_key"]
                    headers["authorization"] = f"Bearer {config['api_key']}"

                if "model" not in payload and "model" in config:
                    payload["model"] = config["model"]
                    
        except Exception as e:
            logger.warning(f"Failed to enrich headers and payload from persistence: {e}")

    async def start(self):
        for name, orch in self.orchestrators.items():
            if orch:
                await orch.start()

    async def stop(self):
        for name, orch in self.orchestrators.items():
            if orch:
                await orch.stop()

    def get_orchestrator(self, headers: Dict[str, str]):
        backend = headers.get("x-backend")
        if not backend:
            raise HTTPException(status_code=400, detail="Missing x-backend header")
            
        backend = backend.lower()
        if backend not in self.orchestrators:
            raise HTTPException(status_code=400, detail=f"Unknown backend {backend}")
            
        orch = self.orchestrators[backend]
        if orch is None:
            raise HTTPException(status_code=400, detail=f"Backend {backend} is disabled on this engine")
            
        return orch

    async def _await_orch_task(self, future_or_coro):
        """Helper to properly await both asyncio coroutines and concurrent futures."""
        if asyncio.iscoroutine(future_or_coro):
            return await future_or_coro
        return await asyncio.wrap_future(future_or_coro)

    async def process_openai_chat(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        messages = payload.get("messages", [])
        is_vision = False
        last_message = messages[-1] if messages else None
        if last_message and isinstance(last_message.get("content"), list):
            if any(p.get("type") == "image_url" for p in last_message["content"]):
                is_vision = True

        task_type = "vision_chat" if is_vision else "text_chat"
        self._enrich_headers_and_payload(task_type, payload, headers)

        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        
        stream = payload.get("stream", False)
        model_name = payload.get("model", "default")
        
        logger.info('received request: %s', json.dumps({
            'task_type': task_type,
            'stream': stream,
            'model_name': model_name,
            'headers': headers
        }))
        
        response = await self._await_orch_task(orch.submit_task(task_type, payload))

        if not stream:
            # 1. Prioridade Alta: Pydantic Models (LiteLLM / API Orchestrator)
            if hasattr(response, 'model_dump_json'):
                return json.loads(response.model_dump_json())
            elif hasattr(response, 'model_dump'):
                return response.model_dump()
            
            # 2. Prioridade Alta: Dicionários completos (Nosso Adapter do GGUF)
            # O usage real e cirúrgico do llama.cpp passa intacto por aqui
            if isinstance(response, dict) and "choices" in response:
                return response
                
            # 3. Fallback (O crime mitigado): Se algum orquestrador simples retornar apenas texto
            if isinstance(response, str):
                logger.warning("Bridge: Orchestrator returned raw string. Estimating tokens.")
                # Estimativa aproximada (1 token ~= 4 chars) para não zerar o painel
                est_tokens = len(response) // 4
    
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": est_tokens, "total_tokens": est_tokens}
                }
            
            return response

        async def openai_generator():
            # Fluxo A: Async Generator Nativo (LiteLLM)
            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    if hasattr(chunk, 'model_dump_json'):
                        chunk_dict = json.loads(chunk.model_dump_json())
                    elif hasattr(chunk, 'model_dump'):
                        chunk_dict = chunk.model_dump()
                    else:
                        chunk_dict = chunk
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                    
            # Fluxo B: Sync Generator (Llama.cpp / GGUF)
            else:
                for chunk in response:
                    if hasattr(chunk, 'model_dump_json'):
                        chunk_dict = json.loads(chunk.model_dump_json())
                    elif hasattr(chunk, 'model_dump'):
                        chunk_dict = chunk.model_dump()
                    else:
                        chunk_dict = chunk
                        
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                    
                    # Cede o controle ao event loop do FastAPI (Crucial para generators síncronos não travarem o servidor)
                    await asyncio.sleep(0) 
                    
            yield "data: [DONE]\n\n"

        return openai_generator()
    
    async def process_anthropic_message(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        messages = payload.get("messages", [])
        is_vision = False
        last_message = messages[-1] if messages else None
        if last_message and isinstance(last_message.get("content"), list):
            if any(p.get("type") == "image" for p in last_message["content"]):
                is_vision = True
        
        task_type = "vision_chat" if is_vision else "text_chat"
        self._enrich_headers_and_payload(task_type, payload, headers)

        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        stream = payload.get("stream", False)
        
        response = await self._await_orch_task(orch.submit_task(task_type, payload))
        
        if not stream:
            # 1. Pydantic Models (LiteLLM nativo Anthropic)
            if hasattr(response, 'model_dump_json'):
                return json.loads(response.model_dump_json())
            elif hasattr(response, 'model_dump'):
                return response.model_dump()
                
            # 2. Dict Completo (Nosso GGUF retornando formato OpenAI)
            # Precisamos converter o dict da OpenAI para o formato Anthropic
            if isinstance(response, dict) and "choices" in response:
                content_text = response["choices"][0].get("message", {}).get("content", "")
                usage = response.get("usage", {})
                return {
                    "id": f"msg_{uuid.uuid4().hex[:10]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": content_text}],
                    "model": payload.get("model", "claude-3"),
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0)
                    }
                }

            # 3. Fallback: String pura
            if isinstance(response, str):
                est_tokens = len(response) // 4
                return {
                    "id": f"msg_{uuid.uuid4().hex[:10]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "model": payload.get("model", "claude-3"),
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 0, "output_tokens": est_tokens}
                }
                
            return response

        async def anthropic_generator():
            msg_id = f"msg_{uuid.uuid4().hex[:10]}"
            model_name = payload.get("model", "claude-3")
            
            # Inicializa a mensagem (Anthropic spec)
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            
            final_output_tokens = 0

            # Função auxiliar para processar cada chunk no padrão OpenAI -> Anthropic
            def process_chunk(chunk_dict):
                nonlocal final_output_tokens
                events = []
                
                # É um chunk normal de texto?
                if 'choices' in chunk_dict and len(chunk_dict['choices']) > 0:
                    delta = chunk_dict['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        events.append(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n")
                
                # É o chunk final contendo o USAGE da OpenAI?
                if 'usage' in chunk_dict:
                     final_output_tokens = chunk_dict['usage'].get('completion_tokens', 0)
                     
                return events

            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk)
                    for event in process_chunk(chunk_dict):
                        yield event
            else:
                for chunk in response:
                    chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk)
                    for event in process_chunk(chunk_dict):
                        yield event
                    await asyncio.sleep(0)
            
            # Encerramento do Stream (Anthropic spec) com os tokens corretos
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': final_output_tokens}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return anthropic_generator()
    
    async def process_embeddings(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[List[float]]:
        self._enrich_headers_and_payload("embedding", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        
        response = await self._await_orch_task(orch.submit_task("embedding", payload))
        
        if isinstance(response, list) and len(response) > 0 and not isinstance(response[0], list):
            return [response]
        return response

    async def process_tts(self, payload: Dict[str, Any], headers: Dict[str, str]) -> bytes:
        self._enrich_headers_and_payload("tts", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        return await self._await_orch_task(orch.submit_task("tts", payload))

    async def process_stt(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        self._enrich_headers_and_payload("stt", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        return await self._await_orch_task(orch.submit_task("stt", payload))

    async def process_image_edit(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        self._enrich_headers_and_payload("pix2pix", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        return await self._await_orch_task(orch.submit_task("pix2pix", payload))

    async def process_image_generation(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        self._enrich_headers_and_payload("image_generation", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        return await self._await_orch_task(orch.submit_task("image_generation", payload))

    async def process_search_web(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Any:
        self._enrich_headers_and_payload("search_web", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        return await self._await_orch_task(orch.submit_task("search_web", payload))

    async def process_search_code(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Any:
        self._enrich_headers_and_payload("search_code", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        return await self._await_orch_task(orch.submit_task("search_code", payload))

bridge = Bridge()
