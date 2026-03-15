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
    and normalizes OpenAI/Anthropic formats returning pure Python dictionaries.
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
                
                has_auth = "authorization" in headers or "x-api-key" in headers
            
                if not has_auth and "api_key" in config and config["api_key"]:
                    headers["x-api-key"] = config["api_key"]
                    headers["authorization"] = f"Bearer {config['api_key']}"

                payload["model"] = config.get("model")
                    
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
        backend = headers.get("x-backend", 'api')
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
        if asyncio.iscoroutine(future_or_coro):
            return await future_or_coro
        return await asyncio.wrap_future(future_or_coro)

    # --- NORMALIZADOR UNIVERSAL ---
    def _normalize_object(self, obj: Any) -> Dict[str, Any]:
        """Extrai o dicionário nativo de qualquer modelo Pydantic ou retorna as-is se já for dict."""
        if hasattr(obj, 'model_dump'):
            # exclude_unset evita que campos None desnecessários quebrem a estrutura
            return obj.model_dump(exclude_unset=True)
        return obj

    async def process_openai_chat(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        messages = payload.get("messages", [])
        last_message = messages[-1] if messages else None
        
        orch = self.get_orchestrator(headers)

        if last_message and isinstance(last_message.get("content"), list):
            new_content = []
            for item in last_message["content"]:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "image_url":
                        new_content.append(item)
                        try:
                            vision_payload = {
                                "image": item["image_url"]["url"],
                                "prompt": "Describe this image in detail.",
                                "model": "default"
                            }
                            vision_res = await self._await_orch_task(orch.submit_task("vision_chat", vision_payload))
                            
                            analysis = ""
                            if isinstance(vision_res, dict) and "choices" in vision_res:
                                analysis = vision_res["choices"][0].get("message", {}).get("content", "")
                            elif isinstance(vision_res, str):
                                analysis = vision_res
                                
                            if analysis:
                                new_content.append({"type": "text", "text": f"User sent an image: {analysis}"})
                        except Exception as e:
                            logger.error(f"Failed to analyze image inline: {e}")
                    elif item_type in ["audio_url", "input_audio"]:
                        new_content.append(item)
                        try:
                            if item_type == "input_audio":
                                b64_audio = item.get("input_audio", {}).get("data", "")
                            else:
                                b64_audio = item.get("audio_url", {}).get("url", "").split(",")[-1]
                            
                            import base64
                            audio_bytes = base64.b64decode(b64_audio)
                            stt_payload = {"audio": audio_bytes, "model": "default"}
                            
                            transcription = await self._await_orch_task(orch.submit_task("stt", stt_payload))
                            if transcription:
                                if isinstance(transcription, dict) and "text" in transcription:
                                    text_val = transcription["text"]
                                else:
                                    text_val = str(transcription)
                                new_content.append({"type": "text", "text": text_val})
                        except Exception as e:
                            logger.error(f"Failed to transcribe audio inline: {e}")
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            
            last_message["content"] = new_content

        task_type = "text_chat"
        self._enrich_headers_and_payload(task_type, payload, headers)

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

        if not response:
            raise HTTPException(status_code=500, detail="Orchestrator returned an empty response")

        if not stream:
            # 1. Normaliza imediatamente (LiteLLM ou GGUF)
            normalized_response = self._normalize_object(response)
            
            # 2. Verifica se é um dicionário OpenAI válido
            if isinstance(normalized_response, dict) and "choices" in normalized_response:
                return normalized_response
                
            # 3. Fallback: Se algum orquestrador retornou apenas texto cru
            logger.warning("Bridge: Orchestrator returned raw string/invalid dict. Estimating tokens.")
            res_content = normalized_response if isinstance(normalized_response, str) else json.dumps(normalized_response)
            est_tokens = len(res_content) // 4
    
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": res_content},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": est_tokens, "total_tokens": est_tokens}
            }

        async def openai_generator():
            # Fluxo Async/Sync Híbrido: Gera apenas Dicionários Puros
            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    yield self._normalize_object(chunk)
            else:
                for chunk in response:
                    yield self._normalize_object(chunk)
                    await asyncio.sleep(0) # Libera o event loop

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
            normalized_response = self._normalize_object(response)
            
            # Converter OpenAI format para Anthropic format
            if isinstance(normalized_response, dict) and "choices" in normalized_response:
                content_text = normalized_response["choices"][0].get("message", {}).get("content", "")
                usage = normalized_response.get("usage", {})
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

            if isinstance(normalized_response, str):
                est_tokens = len(normalized_response) // 4
                return {
                    "id": f"msg_{uuid.uuid4().hex[:10]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": normalized_response}],
                    "model": payload.get("model", "claude-3"),
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 0, "output_tokens": est_tokens}
                }
                
            return normalized_response

        async def anthropic_generator():
            msg_id = f"msg_{uuid.uuid4().hex[:10]}"
            model_name = payload.get("model", "claude-3")
            
            # Inicializa a mensagem (Anthropic spec) - Mantido como String SSE pois o padrão Anthropic exige os blocos "event:"
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            
            final_output_tokens = 0

            def process_chunk(chunk_dict):
                nonlocal final_output_tokens
                events = []
                
                if 'choices' in chunk_dict and len(chunk_dict['choices']) > 0:
                    delta = chunk_dict['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        events.append(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n")
                
                if 'usage' in chunk_dict and chunk_dict['usage']:
                     final_output_tokens = chunk_dict['usage'].get('completion_tokens', 0)
                     
                return events

            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    for event in process_chunk(self._normalize_object(chunk)):
                        yield event
            else:
                for chunk in response:
                    for event in process_chunk(self._normalize_object(chunk)):
                        yield event
                    await asyncio.sleep(0)
            
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