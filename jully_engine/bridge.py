import asyncio
import base64
import io
import logging
import uuid
import time
import json
from fastapi import HTTPException
from typing import Any, Dict, Optional, Union, AsyncGenerator, List

from .events import event_manager
from .services.helpers import inference_helper, MultiModalHelper

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
    async def start(self):
        # O container agora guarda as instâncias dos orquestradores
        for name, orch in inference_helper.orchestrator_container.orchestrators.items():
            if orch:
                await orch.start()

    async def stop(self):
        for name, orch in inference_helper.orchestrator_container.orchestrators.items():
            if orch:
                await orch.stop()

    def _normalize_object(self, obj: Any) -> Any:
        """
        Desmonta modelos Pydantic (LiteLLM/Llama.cpp) e classes genéricas
        em dicionários puros de Python, garantindo o JSON Serializable.
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
            
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return self._normalize_object(obj.model_dump())
            
        if hasattr(obj, "dict") and callable(obj.dict):
            return self._normalize_object(obj.dict())
            
        if isinstance(obj, dict):
            return {k: self._normalize_object(v) for k, v in obj.items()}
            
        if isinstance(obj, (list, tuple)):
            return [self._normalize_object(i) for i in obj]
            
        if hasattr(obj, "__dict__"):
            return self._normalize_object(vars(obj))
            
        # Handle NumPy types if present
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return self._normalize_object(obj.tolist())
            if isinstance(obj, np.generic):
                return obj.item()
        except ImportError:
            pass
            
        return str(obj)
    
    async def process_openai_chat(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        start_time = time.time()
        task_type = "text_chat"
        payload['headers'] = headers

        try:
            helper = MultiModalHelper(payload=payload)
            await helper.process_transcription()
            await helper.process_vision()
            
            messages = payload.get("messages", [])
            input_chars = len(json.dumps(messages)) if messages else 0
        except Exception as e:
            logger.error(f"Multimodal Helper failed: {e}")
            messages = payload.get("messages", [])
            input_chars = len(json.dumps(messages)) if messages else 0

        stream = payload.get("stream", False)
        
        logger.info("Text request received: " + json.dumps({
            "headers": headers,
            "model": payload.get("model", "default"),
            "stream": stream
        }))
        
        response = await inference_helper.process(task_type, payload)
        
        if not response:
            raise HTTPException(status_code=500, detail="Orchestrator returned an empty response")

        model_name = payload.get("model", "default")

        if not stream:
            normalized_response = self._normalize_object(response)
            gen_time = time.time() - start_time
            
            tokens = normalized_response.get("usage", {}).get("total_tokens", 0) if isinstance(normalized_response, dict) else 0
            interaction_id = normalized_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:10]}") if isinstance(normalized_response, dict) else f"chatcmpl-{uuid.uuid4().hex[:10]}"
            
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

        async def openai_generator():
            tokens = 0
            interaction_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            try:
                iterator = response if hasattr(response, '__aiter__') else response
                
                async for chunk in iterator:
                    normalized = self._normalize_object(chunk)
                    if isinstance(normalized, dict):
                        if "id" in normalized:
                            interaction_id = normalized["id"]
                        if "usage" in normalized and normalized["usage"]:
                            tokens = normalized["usage"].get("total_tokens", tokens)
                        
                        # Normalização Universal de Reasoning
                        if "choices" in normalized and len(normalized["choices"]) > 0:
                            delta = normalized["choices"][0].get("delta", {})
                            # Padrão LiteLLM às vezes usa 'reasoning'
                            if "reasoning" in delta and "reasoning_content" not in delta:
                                delta["reasoning_content"] = delta.pop("reasoning")
                            # Alguns provedores usam 'thought'
                            if "thought" in delta and "reasoning_content" not in delta:
                                delta["reasoning_content"] = delta.pop("thought")
                                
                    yield normalized
                    await asyncio.sleep(0)
            finally:
                gen_time = time.time() - start_time
                event_manager.emit(task_type, tokens_spent=tokens, generation_time=gen_time, input_chars=input_chars, interaction_id=interaction_id)

        return openai_generator()
    
    async def process_anthropic_message(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        # 1. ADAPTER IN: Convertendo Payload Anthropic para OpenAI
        openai_payload = {"model": payload.get("model", "claude-3"), "stream": payload.get("stream", False)}
        
        for key in ["temperature", "max_tokens", "top_p", "top_k"]:
            if key in payload:
                openai_payload[key] = payload[key]

        openai_messages = []
        
        if "system" in payload:
            system_content = payload["system"]
            if isinstance(system_content, list):
                system_text = "".join(b.get("text", "") for b in system_content if b.get("type") == "text")
                openai_messages.append({"role": "system", "content": system_text})
            else:
                openai_messages.append({"role": "system", "content": str(system_content)})

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
        stream = openai_payload.get("stream", False)
        openai_response = await self.process_openai_chat(openai_payload, headers)
        
        # 3. ADAPTER OUT (Sync)
        if not stream:
            message = openai_response["choices"][0].get("message", {})
            content_text = message.get("content") or ""
            reasoning_text = message.get("reasoning_content") or ""
            
            usage = openai_response.get("usage", {})
            interaction_id = openai_response.get("id", f"msg_{uuid.uuid4().hex[:10]}")
            
            res = {
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
            
            if reasoning_text:
                res["reasoning_content"] = reasoning_text
                
            return res

        # 4. ADAPTER OUT (Async - SSE Stream Anthropic)
        async def anthropic_generator():
            msg_id = f"msg_{uuid.uuid4().hex[:10]}"
            model_name = openai_payload.get("model", "claude-3")
            
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            
            final_output_tokens = 0
            
            async for openai_chunk in openai_response:
                if 'choices' in openai_chunk and len(openai_chunk['choices']) > 0:
                    delta = openai_chunk['choices'][0].get('delta', {})
                    
                    if 'reasoning_content' in delta and delta['reasoning_content']:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['reasoning_content']}})}\n\n"
                    
                    if 'content' in delta and delta['content']:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                
                if 'usage' in openai_chunk and openai_chunk['usage']:
                     final_output_tokens = openai_chunk['usage'].get('completion_tokens', final_output_tokens)
                     
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': final_output_tokens}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return anthropic_generator()
    
    async def process_embeddings(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[List[float]]:
        payload['headers'] = headers
        response = await inference_helper.process("embeddings", payload)
        
        if isinstance(response, list) and len(response) > 0 and not isinstance(response[0], list):
            return [response]
        return response

    async def process_tts(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Union[bytes, AsyncGenerator[bytes, None]]:
        start_time = time.time()
        input_chars = len(payload.get("input", ""))
        payload['headers'] = headers
        stream = payload.get("stream", False)
        
        audio_result = await inference_helper.process("tts", payload)
        gen_time = time.time() - start_time
        
        if stream and hasattr(audio_result, '__aiter__'):
            event_manager.emit("voice", generation_time=gen_time, input_chars=input_chars, audio_duration=0.0)
            return audio_result
        
        audio_duration = get_audio_duration(audio_result) if audio_result else 0.0
        event_manager.emit("voice", generation_time=gen_time, input_chars=input_chars, audio_duration=audio_duration)
        return audio_result

    async def process_stt(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        start_time = time.time()
        audio_bytes = payload.get("audio", b"")
        audio_duration = get_audio_duration(audio_bytes) if audio_bytes else 0.0
        payload['headers'] = headers
        
        result = await inference_helper.process("stt", payload)
        
        gen_time = time.time() - start_time
        event_manager.emit("stt", generation_time=gen_time, audio_duration=audio_duration)
        return result

    async def process_image_edit(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        start_time = time.time()
        payload['headers'] = headers
        
        # OpenAI Compatibility: Parse size (WxH)
        if "size" in payload and payload["size"]:
            try:
                w, h = payload["size"].lower().split("x")
                payload["width"] = int(w)
                payload["height"] = int(h)
            except Exception:
                logger.warning(f"Bridge: Failed to parse size '{payload['size']}'")
        
        n = max(1, int(payload.get("n", 1)))
        response_format = payload.get("response_format", "b64_json")
        
        results = []
        for i in range(n):
            if n > 1:
                logger.info(f"Bridge: Generating image {i+1}/{n} (Edit)")
            res = await inference_helper.process("pix2pix", payload)
            results.append(res)
        
        gen_time = time.time() - start_time
        event_manager.emit("image", generation_time=gen_time)
        
        data = []
        for res in results:
            if response_format == "b64_json":
                data.append({"b64_json": res})
            else:
                data.append({"url": f"data:image/png;base64,{res}"})
                
        return {
            "created": int(time.time()),
            "data": data
        }

    async def process_image_resize(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        start_time = time.time()
        payload['headers'] = headers
        
        result = await inference_helper.process("image_resize", payload)
        
        gen_time = time.time() - start_time
        event_manager.emit("image", generation_time=gen_time)
        return result

    async def process_image_generation(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        start_time = time.time()
        payload['headers'] = headers
        
        # OpenAI Compatibility: Parse size (WxH)
        if "size" in payload and payload["size"]:
            try:
                w, h = payload["size"].lower().split("x")
                payload["width"] = int(w)
                payload["height"] = int(h)
            except Exception:
                logger.warning(f"Bridge: Failed to parse size '{payload['size']}'")

        n = max(1, int(payload.get("n", 1)))
        response_format = payload.get("response_format", "b64_json")
        
        results = []
        for i in range(n):
            if n > 1:
                logger.info(f"Bridge: Generating image {i+1}/{n} (Generation)")
            res = await inference_helper.process("image_generation", payload)
            results.append(res)
        
        gen_time = time.time() - start_time
        event_manager.emit("image", generation_time=gen_time)
        
        data = []
        for res in results:
            if response_format == "b64_json":
                data.append({"b64_json": res})
            else:
                data.append({"url": f"data:image/png;base64,{res}"})
                
        return {
            "created": int(time.time()),
            "data": data
        }

    async def process_search_web(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Any:
        start_time = time.time()
        headers['x-backend'] = 'api'
        payload['headers'] = headers
        
        res = await inference_helper.process("search_web", payload)
        
        gen_time = time.time() - start_time
        event_manager.emit("search_web", generation_time=gen_time)
        return res

    async def process_search_code(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Any:
        start_time = time.time()
        headers['x-backend'] = 'api'
        payload['headers'] = headers
        
        res = await inference_helper.process("search_code", payload)
        
        gen_time = time.time() - start_time
        event_manager.emit("search_code", generation_time=gen_time)
        return res

    async def process_video_description(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        from .model_loader import model_loader

        payload['headers'] = headers
        
        backend, model = inference_helper.orchestrator_container.resolve_backend('VISION', payload)
        
        return await model_loader.get_eyes(backend, model).describe_video(payload)

    async def process_face_sync_batch(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[List[Dict[str, Any]]]:
        """Sincroniza rostos de múltiplas imagens de uma única vez."""
        from PIL import Image
        from .model_loader import model_loader
        
        backend, model = inference_helper.orchestrator_container.resolve_backend('VISION', payload)
        eyes = model_loader.get_eyes(backend, model)
        
        collection = payload.get("collection", "faces_embeddings")
        images_b64 = payload.get("images", [])
        pic_ids = payload.get("pic_ids", [])
        
        images = []
        for img_b64 in images_b64:
            clean_b64 = img_b64.split(",")[1] if img_b64.startswith("data:") else img_b64
            img_data = base64.b64decode(clean_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
            
        # Garante que pic_ids tenha o mesmo tamanho que images
        if len(pic_ids) < len(images):
            pic_ids += ["unknown"] * (len(images) - len(pic_ids))
            
        return await eyes.sync_faces_batch(images, pic_ids, collection)

    async def process_face_extraction(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[Dict[str, Any]]:
        from PIL import Image
        from .model_loader import model_loader
        
        backend, model = inference_helper.orchestrator_container.resolve_backend('VISION', payload)
        eyes = model_loader.get_eyes(backend, model)
        
        # 1. Extrai a coleção (default faces_embeddings)
        collection = payload.get("collection", "faces_embeddings")

        # 2. Decodifica as imagens puras
        images_b64 = payload.get("images", [])
        images = []
  
        for img_b64 in images_b64:
            # Remove o prefixo data URI caso o cliente tenha mandado
            clean_b64 = img_b64.split(",")[1] if img_b64.startswith("data:") else img_b64
            img_data = base64.b64decode(clean_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
            
        return await eyes.describe_person_faces(images, collection=collection)

    async def process_image_description(self, payload: Dict[str, Any], headers: Dict[str, str]) -> List[str]:
        # 1. O Bridge é burro. Ele não instancia classes, ele só formata o pacote.
        images = payload.get("images", [])
        prompt = payload.get("prompt", "Describe these images in detail.")
        model = payload.get("model", None)
        
        vision_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # Adiciona todas as imagens formatadas no padrão OpenAI
                        *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images],
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "headers": headers
        }
        
        results = await inference_helper.process('vision_chat', vision_payload)
        
        if isinstance(results, list):
            return results

        return [str(results)]

    async def process_rag_add(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Adiciona um texto ao banco vetorial via Memory domain mapeado no inference_helper."""
        payload['headers'] = headers
        success = await inference_helper.process("rag_add", payload)
        return {"success": success, "collection": payload.get("collection", "july_memory")}

    async def process_rag_batch_add(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Insere múltiplos documentos no RAG orquestrado."""
        payload['headers'] = headers
        result = await inference_helper.process("rag_batch_add", payload)
        return result

    async def process_rag_search(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Busca contexto no RAG orquestrado."""
        payload['headers'] = headers
        result = await inference_helper.process("rag_search", payload)
        normalized = self._normalize_object(result)
        return {"results": normalized, "collection": payload.get("collection", "july_memory")}

    async def process_rag_vector_add(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Adiciona um vetor bruto diretamente ao banco orquestrado."""
        payload['headers'] = headers
        success = await inference_helper.process("rag_vector_add", payload)
        return {"success": success, "collection": payload.get("collection", "july_memory")}


    async def process_rag_update(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Atualiza a coordenada geométrica de um vetor orquestrado."""
        payload['headers'] = headers
        success = await inference_helper.process("rag_update", payload)
        return {"success": success}

    async def process_rag_delete(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Remove registros do RAG orquestrado."""
        payload['headers'] = headers
        count = await inference_helper.process("rag_delete", payload)
        return {"deleted_count": count}

    async def process_rag_list(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Lista metadados de uma coleção orquestrada."""
        payload['headers'] = headers
        results = await inference_helper.process("rag_list", payload)
        return {"results": results}

    async def process_rag_smart_search(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Realiza busca inteligente orquestrada no RAG."""
        payload['headers'] = headers
        results = await inference_helper.process("rag_smart_search", payload)
        return {"results": results}

bridge = Bridge()