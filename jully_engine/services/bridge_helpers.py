import io
import json
import base64
import uuid
import logging
import asyncio
from typing import Any, Dict, List, AsyncGenerator

logger = logging.getLogger("JulyEngine.Services.BridgeHelpers")

class AudioUtils:
    @staticmethod
    def get_duration(audio_bytes: bytes) -> float:
        if not audio_bytes: return 0.0
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

class ResponseNormalizer:
    @classmethod
    def normalize(cls, obj: Any) -> Any:
        """Desmonta modelos Pydantic e classes em dicionários puros."""
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return cls.normalize(obj.model_dump())
        if hasattr(obj, "dict") and callable(obj.dict):
            return cls.normalize(obj.dict())
        if isinstance(obj, dict):
            return {k: cls.normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls.normalize(i) for i in obj]
        if hasattr(obj, "__dict__"):
            return cls.normalize(vars(obj))
        return str(obj)

class PayloadEnricher:
    @staticmethod
    def enrich(task_key: str, payload: Dict[str, Any], headers: Dict[str, str]):
        try:
            from ..persistence import get_backend
            backend_db = get_backend()
            config = None
            model_alias = payload.get("model")

            if task_key in ["text_chat"]:
                text_presets = backend_db.get_setting("TEXT_PRESETS") or []
                config = next((p for p in text_presets if p.get("alias") == model_alias), None)
                if not config and text_presets:
                    config = text_presets[0]

            if not config:
                mapping = {
                    "tts": "TTS", "stt": "STT", "vision_chat": "VISION",
                    "embeddings": "EMBEDDINGS", "pix2pix": "IMAGE_EDIT",
                    "image_generation": "IMAGE_CREATE", "search_web": "WEB_SEARCH",
                    "search_code": "REPOSITORY_SEARCH"
                }
                setting_key = mapping.get(task_key)
                if setting_key:
                    config = backend_db.get_setting(setting_key)

            if not config and task_key == "vision_chat" and model_alias in ["fastvlm", "moondream"]:
                headers.setdefault("x-backend", "gpu")

            if config:
                headers.setdefault("x-backend", config.get('backend', 'gpu'))
                if not payload.get("model"):
                    payload["model"] = config.get('model', 'default')

        except Exception as e:
            logger.warning(f"Failed to enrich backend: {e}")

class AnthropicAdapter:
    @staticmethod
    def convert_in(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Converte request Anthropic para padrão OpenAI"""
        openai_payload = {"model": payload.get("model", "claude-3"), "stream": payload.get("stream", False)}
        for key in ["temperature", "max_tokens", "top_p", "top_k"]:
            if key in payload: openai_payload[key] = payload[key]

        openai_messages = []
        if "system" in payload:
            system_content = payload["system"]
            if isinstance(system_content, list):
                system_text = "".join(b.get("text", "") for b in system_content if b.get("type") == "text")
                openai_messages.append({"role": "system", "content": system_text})
            else:
                openai_messages.append({"role": "system", "content": str(system_content)})

        for msg in payload.get("messages", []):
            role, content = msg.get("role"), msg.get("content")
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
                        openai_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{source.get('data', '')}"}})
                openai_messages.append({"role": role, "content": openai_content})
        
        openai_payload["messages"] = openai_messages
        return openai_payload

    @staticmethod
    def convert_out_sync(openai_response: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Converte resposta síncrona OpenAI para Anthropic"""
        message = openai_response["choices"][0].get("message", {})
        usage = openai_response.get("usage", {})
        res = {
            "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:10]}"),
            "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": message.get("content") or ""}],
            "model": model_name, "stop_reason": "end_turn",
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        }
        if message.get("reasoning_content"):
            res["reasoning_content"] = message.get("reasoning_content")
        return res

    @staticmethod
    async def convert_out_stream(openai_response: AsyncGenerator, model_name: str) -> AsyncGenerator[str, None]:
        """Converte stream OpenAI para SSE Anthropic"""
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        final_output_tokens = 0
        async for chunk in openai_response:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                for key in ['reasoning_content', 'content']:
                    if key in delta and delta[key]:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta[key]}})}\n\n"
            if 'usage' in chunk and chunk['usage']:
                 final_output_tokens = chunk['usage'].get('completion_tokens', final_output_tokens)
                 
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': final_output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

class MultimodalHandler:
    """Intercepa e processa áudio e imagem antes de enviar ao modelo de texto."""
    def __init__(self, bridge):
        self.bridge = bridge

    async def process(self, payload: Dict[str, Any], headers: Dict[str, str]):
        messages = payload.get("messages", [])
        if not messages: return

        last_message = messages[-1]
        if not isinstance(last_message.get("content"), list): return

        new_content, image_items = [], []

        for item in last_message["content"]:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "image_url":
                    image_items.append(item)
                elif item_type in ["audio_url", "input_audio"]:
                    b64_audio = item.get("input_audio", {}).get("data", "") if item_type == "input_audio" else item.get("audio_url", {}).get("url", "").split(",")[-1]
                    stt_payload = {"audio": base64.b64decode(b64_audio), "model": "default"}
                    stt_headers = {}
                    PayloadEnricher.enrich('stt', stt_payload, stt_headers)
                    stt_orch = self.bridge.get_orchestrator(stt_headers)
                    
                    try:
                        transcription = await self.bridge._await_orch_task(stt_orch.submit_task("stt", stt_payload))
                        text_val = transcription.get("text", str(transcription)) if isinstance(transcription, dict) else str(transcription)
                        new_content.append({"type": "text", "text": f"[User sent audio]: {text_val}"})
                    except Exception as e:
                        logger.error(f"Multimodal STT failed: {e}")
                else:
                    new_content.append(item)
            else:
                new_content.append(item)

        if image_items:
            try:
                images_to_analyze = [item["image_url"]["url"] for item in image_items]
                vision_payload = {"images": images_to_analyze, "prompt": "Describe this image in detail."}
                vision_headers = {}
                PayloadEnricher.enrich('vision_chat', vision_payload, vision_headers)
                vision_orch = self.bridge.get_orchestrator(vision_headers)
                
                vision_res = await self.bridge._await_orch_task(vision_orch.submit_task("vision_chat", vision_payload))
                
                batch_analyses = []
                if isinstance(vision_res, list):
                    batch_analyses = [r.get("choices", [{}])[0].get("message", {}).get("content", str(r)) if isinstance(r, dict) else str(r) for r in vision_res]
                elif isinstance(vision_res, dict) and "choices" in vision_res:
                    batch_analyses = [vision_res["choices"][0].get("message", {}).get("content", "")] * len(image_items)

                for i, desc in enumerate(batch_analyses):
                    # BUG CORRIGIDO: Agora NÃO adicionamos a imagem original de volta, apenas a descrição em texto!
                    new_content.append({"type": "text", "text": f"[User attached an image. Description provided by Vision model: {desc}]"})
            except Exception as e:
                logger.error(f"Multimodal Vision failed: {e}")

        last_message["content"] = new_content