import asyncio
import logging
import uuid
import time
import json
import os
from fastapi import HTTPException
from typing import Any, Dict, Optional, Union, AsyncGenerator, List
from .orchestrators.api_orchestrator import api_orchestrator
from .events import event_manager

logger = logging.getLogger("JulyEngine.Bridge")

def get_audio_duration(audio_bytes: bytes) -> float:
    import io
    try:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            return len(f) / f.samplerate
    except Exception:
        import wave
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as f:
                return f.getnframes() / float(f.getframerate())
        except Exception:
            return 0.0

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

                if "model" in config and not payload.get("model"):
                    payload["model"] = config["model"]
                    
        except Exception as e:
            logger.warning(f"Failed to enrich backend from persistence: {e}")

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

    def _normalize_object(self, obj: Any) -> Any:
        """
        Desmonta modelos Pydantic (LiteLLM/Llama.cpp) e classes genéricas
        em dicionários puros de Python, garantindo o JSON Serializable.
        """
        # Caminho rápido para primitivos
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
            
        # Pydantic v2
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return self._normalize_object(obj.model_dump())
            
        # Pydantic v1 / Objetos nativos do LiteLLM (como ModelResponse)
        if hasattr(obj, "dict") and callable(obj.dict):
            return self._normalize_object(obj.dict())
            
        # Se for um dicionário, normaliza os valores recursivamente
        if isinstance(obj, dict):
            return {k: self._normalize_object(v) for k, v in obj.items()}
            
        # Se for lista ou tupla (como a lista de 'choices')
        if isinstance(obj, (list, tuple)):
            return [self._normalize_object(i) for i in obj]
            
        # Se for uma classe qualquer do Python (como StreamingChoices)
        if hasattr(obj, "__dict__"):
            return self._normalize_object(vars(obj))
            
        # Fallback de segurança
        return str(obj)
    
    async def process_openai_chat(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        start_time = time.time()
        messages = payload.get("messages", [])
        input_chars = len(json.dumps(messages)) if messages else 0
        last_message = messages[-1] if messages else None
        task_type = "text_chat"
        self._enrich_headers_and_payload(task_type, payload, headers)
        
        orch = self.get_orchestrator(headers)

        # 1. PROCESSAMENTO INLINE DE MÍDIA (Visão/Áudio) - Refatorado para BATCHING
        if last_message and isinstance(last_message.get("content"), list):
            new_content = []
            image_items = []
            
            for item in last_message["content"]:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "image_url":
                        image_items.append(item)
                    elif item_type in ["audio_url", "input_audio"]:
                        # Áudio continua um por um por enquanto (latência menor individualmente)
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
                                text_val = transcription["text"] if isinstance(transcription, dict) and "text" in transcription else str(transcription)
                                new_content.append({"type": "text", "text": text_val})
                        except Exception as e:
                            logger.error(f"Failed to transcribe audio inline: {e}")
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)

            # Processamento em BATCH de imagens
            if image_items:
                try:
                    images_to_analyze = [item["image_url"]["url"] for item in image_items]
                    vision_payload = {
                        "images": images_to_analyze, # Nota: mudamos de 'image' para 'images'
                        "prompt": "Describe this image in detail.",
                        "model": "default"
                    }
                    
                    # O orquestrador deve lidar com a lista de imagens
                    vision_res = await self._await_orch_task(orch.submit_task("vision_chat", vision_payload))
                    
                    # Mapear os resultados de volta para o conteúdo
                    # Se vier uma lista de strings ou lista de objetos OpenAI
                    batch_analyses = []
                    if isinstance(vision_res, list):
                        for r in vision_res:
                            if isinstance(r, dict) and "choices" in r:
                                batch_analyses.append(r["choices"][0].get("message", {}).get("content", ""))
                            else:
                                batch_analyses.append(str(r))
                    elif isinstance(vision_res, dict) and "choices" in vision_res:
                        # Se o orquestrador retornar apenas uma resposta (talvez concatenada ou erro de batching)
                        batch_analyses = [vision_res["choices"][0].get("message", {}).get("content", "")] * len(image_items)
                    
                    for i, item in enumerate(image_items):
                        new_content.append(item)
                        if i < len(batch_analyses) and batch_analyses[i]:
                            new_content.append({"type": "text", "text": f"User sent an image: {batch_analyses[i]}"})
                except Exception as e:
                    logger.error(f"Failed to analyze images in batch: {e}")
                    # Fallback: re-adicionar as imagens sem análise se falhar
                    for item in image_items:
                        new_content.append(item)

            last_message["content"] = new_content

        # 2. CONFIGURAÇÃO DO PAYLOAD E CHAMADA
        payload['headers'] = headers
        stream = payload.get("stream", False)
        model_name = payload.get("model", "default")
        
        logger.info("Text request received: " + json.dumps({
            "headers": headers,
            "model": model_name,
            "stream": stream
        }))
        
        response = await self._await_orch_task(orch.submit_task(task_type, payload))
        if not response:
            raise HTTPException(status_code=500, detail="Orchestrator returned an empty response")

        # 3. RETORNO SÍNCRONO (Limpo, preservando a estrutura original do LiteLLM)
        if not stream:
            normalized_response = self._normalize_object(response)
            gen_time = time.time() - start_time
            
            # Pega o usage e id diretamente do dicionário normalizado, se existirem
            tokens = normalized_response.get("usage", {}).get("total_tokens", 0) if isinstance(normalized_response, dict) else 0
            interaction_id = normalized_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:10]}") if isinstance(normalized_response, dict) else f"chatcmpl-{uuid.uuid4().hex[:10]}"
            
            # Fallback para string crua (caso extremo)
            if isinstance(normalized_response, str):
                est_tokens = len(normalized_response) // 4
                event_manager.emit(task_type, tokens_spent=est_tokens, generation_time=gen_time, input_chars=input_chars, interaction_id=interaction_id)
                return {
                    "id": interaction_id, "object": "chat.completion", "created": int(time.time()), "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": normalized_response}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": est_tokens, "total_tokens": est_tokens}
                }
            
            event_manager.emit(task_type, tokens_spent=tokens, generation_time=gen_time, input_chars=input_chars, interaction_id=interaction_id)
            return normalized_response

        # 4. RETORNO ASSÍNCRONO (STREAM)
        async def openai_generator():
            tokens = 0
            interaction_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            try:
                # O LiteLLM/Llama.cpp já gerenciam o iterador corretamente.
                iterator = response if hasattr(response, '__aiter__') else response
                
                async for chunk in iterator:
                    normalized = self._normalize_object(chunk)
                    if isinstance(normalized, dict):
                        if "id" in normalized:
                            interaction_id = normalized["id"]
                        if "usage" in normalized and normalized["usage"]:
                            tokens = normalized["usage"].get("total_tokens", tokens)
                    yield normalized
                    await asyncio.sleep(0)
            finally:
                gen_time = time.time() - start_time
                event_manager.emit(task_type, tokens_spent=tokens, generation_time=gen_time, input_chars=input_chars, interaction_id=interaction_id)

        return openai_generator()
    
    async def process_anthropic_message(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        # 1. ADAPTER IN: Convertendo Payload Anthropic para OpenAI
        openai_payload = {"model": payload.get("model", "claude-3"), "stream": payload.get("stream", False)}
        
        # O Anthropic passa parâmetros adicionais (temperature, max_tokens, etc)
        for key in ["temperature", "max_tokens", "top_p", "top_k"]:
            if key in payload:
                openai_payload[key] = payload[key]

        openai_messages = []
        
        # O Anthropic manda o System Prompt solto na raiz do payload. Convertemos para role: system.
        if "system" in payload:
            system_content = payload["system"]
            # Às vezes o system vem como array de blocos de texto no Anthropic
            if isinstance(system_content, list):
                system_text = "".join(b.get("text", "") for b in system_content if b.get("type") == "text")
                openai_messages.append({"role": "system", "content": system_text})
            else:
                openai_messages.append({"role": "system", "content": str(system_content)})

        # Tradução das mensagens (role: user e role: assistant)
        for msg in payload.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                openai_content = []
                for block in content:
                    if block.get("type") == "text":
                        openai_content.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image":
                        # Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                        source = block.get("source", {})
                        mime = source.get("media_type", "image/jpeg")
                        b64_data = source.get("data", "")
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64_data}"}
                        })
                openai_messages.append({"role": role, "content": openai_content})
                
        openai_payload["messages"] = openai_messages

        # 2. PROCESSAMENTO: Chama o tubo universal da OpenAI
        # Isso já vai lidar com análise inline de imagem, stt, chamadas ao orquestrador e normalização!
        openai_response = await self.process_openai_chat(openai_payload, headers)
        
        stream = openai_payload["stream"]

        # 3. ADAPTER OUT (Sync)
        if not stream:
            # Pega o dicionário validado do OpenAI e transforma no formato Claude
            content_text = openai_response["choices"][0].get("message", {}).get("content", "")
            usage = openai_response.get("usage", {})
            interaction_id = openai_response.get("id", f"msg_{uuid.uuid4().hex[:10]}")
            
            return {
                "id": interaction_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                "model": openai_payload["model"],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0)
                }
            }

        # 4. ADAPTER OUT (Async - SSE Stream Anthropic)
        async def anthropic_generator():
            msg_id = f"msg_{uuid.uuid4().hex[:10]}"
            model_name = openai_payload["model"]
            
            # Handshake inicial do protocolo Anthropic
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            
            final_output_tokens = 0
            
            # openai_response aqui é o async generator retornado por process_openai_chat
            async for openai_chunk in openai_response:
                # O chunk já vem limpo e normalizado pelo process_openai_chat!
                if 'choices' in openai_chunk and len(openai_chunk['choices']) > 0:
                    delta = openai_chunk['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                
                # Se o chunk trouxer o usage final (comum na última iteração do LiteLLM)
                if 'usage' in openai_chunk and openai_chunk['usage']:
                     final_output_tokens = openai_chunk['usage'].get('completion_tokens', final_output_tokens)
                     
            # Encerramento do Stream SSE no padrão Anthropic
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
        start_time = time.time()
        input_chars = len(payload.get("input", ""))
        self._enrich_headers_and_payload("tts", payload, headers)
        
        # Inject defaults from config if not strictly provided
        if not payload.get("voice") or not payload.get("language"):
            from .persistence import get_backend
            
            db = get_backend()
            tts_config = db.get_setting("TTS") or {}
            
            if not payload.get("voice") and tts_config.get("voice"):
                payload["voice"] = tts_config["voice"]
            
            if not payload.get("language") and tts_config.get("language"):
                payload["language"] = tts_config["language"]
                
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        audio_bytes = await self._await_orch_task(orch.submit_task("tts", payload))
        gen_time = time.time() - start_time
        audio_duration = get_audio_duration(audio_bytes) if audio_bytes else 0.0
        event_manager.emit("voice", generation_time=gen_time, input_chars=input_chars, audio_duration=audio_duration)
        return audio_bytes

    async def process_stt(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        start_time = time.time()
        audio_bytes = payload.get("audio", b"")
        audio_duration = get_audio_duration(audio_bytes) if audio_bytes else 0.0
        self._enrich_headers_and_payload("stt", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        result = await self._await_orch_task(orch.submit_task("stt", payload))
        gen_time = time.time() - start_time
        event_manager.emit("stt", generation_time=gen_time, audio_duration=audio_duration)
        return result

    async def process_image_edit(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        start_time = time.time()
        self._enrich_headers_and_payload("pix2pix", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        result = await self._await_orch_task(orch.submit_task("pix2pix", payload))
        gen_time = time.time() - start_time
        event_manager.emit("image", generation_time=gen_time)
        return result

    async def process_image_generation(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        start_time = time.time()
        self._enrich_headers_and_payload("image_generation", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        result = await self._await_orch_task(orch.submit_task("image_generation", payload))
        gen_time = time.time() - start_time
        event_manager.emit("image", generation_time=gen_time)
        return result

    async def process_search_web(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Any:
        start_time = time.time()
        self._enrich_headers_and_payload("search_web", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        res = await self._await_orch_task(orch.submit_task("search_web", payload))
        gen_time = time.time() - start_time
        event_manager.emit("search_web", generation_time=gen_time)
        return res

    async def process_search_code(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Any:
        start_time = time.time()
        self._enrich_headers_and_payload("search_code", payload, headers)
        orch = self.get_orchestrator(headers)
        payload['headers'] = headers
        res = await self._await_orch_task(orch.submit_task("search_code", payload))
        gen_time = time.time() - start_time
        event_manager.emit("search_code", generation_time=gen_time)
        return res

bridge = Bridge()