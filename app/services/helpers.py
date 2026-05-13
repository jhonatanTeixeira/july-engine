import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MultiModalHelper:
    payload: Dict
    
    def __post_init__(self):
        self.messages = self.payload.get("messages")
        
    def _get_multimodal_content(self, messages: Optional[List[Dict]] = None):
        target_messages = messages if messages is not None else self.messages
        if not target_messages: return
        
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
        
    def get_texts(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return [
            part for part in self._get_multimodal_content(messages) 
            if part.get("type", "text") == "text"
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
        if not msgs: return []
        
        for msg in msgs:
            content = msg.get("content", "")
            
            if isinstance(content, list):
                msg["content"] = [part for part in content if part.get("type") not in types_to_remove]
        
        return msgs

    def filter_images(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["image_url"], messages)
        
    def filter_audios(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["audio_url", "input_audio"], messages)

    def flatten_messages(self, messages: List[Dict]) -> List[Dict]:
        """Converte mensagens com conteúdo em lista para string simples, removendo partes não textuais."""
        new_messages = []
        for msg in messages:
            new_msg = deepcopy(msg)
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                new_msg["content"] = "\n".join(text_parts)
            else:
                new_msg["content"] = str(content)
            
            new_messages.append(new_msg)
        return new_messages

    def last_message_images(self) -> List[Dict]:
        if not self.messages: return []
        return self.get_images([self.messages[-1]])
        
    def last_message_text(self) -> str:
        if not self.messages: return ""
        return '\n'.join([t['text'] for t in self.get_texts([self.messages[-1]])])

    def last_message_audios(self) -> List[Dict]:
        if not self.messages: return []
        return self.get_audios([self.messages[-1]])

    async def _await_orch_task(self, future_or_coro):
        if asyncio.iscoroutine(future_or_coro):
            res = await future_or_coro
        else:
            res = await asyncio.wrap_future(future_or_coro)
        
        if asyncio.iscoroutine(res):
            return await res

        return res

    async def process_vision(self):
        from ..bridge import bridge
        from ..services.models_service import model_service
        

        if (last_images := self.last_message_images()):
            vision_content = deepcopy(last_images) 
            vision_content.append({
                "type": "text", 
                "text": 'describe this image in detail'
            })

            model = (model_service.backend.get_setting("VISION") or {}).get("model", 'fastvlm')

            vision_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": vision_content
                    }
                ]
            }

            headers = {}

            response: List[str] = await bridge.process_image_description(vision_payload, headers)
            
            # Garante que o conteúdo seja uma lista para podermos adicionar a descrição
            current_content = self.messages[-1].get("content", "")
            if isinstance(current_content, str):
                self.messages[-1]["content"] = [{"type": "text", "text": current_content}]
            
            last_content: list = self.messages[-1]["content"]
            
            format_content = lambda content: {"type": "text", "text": f"[User sent an image]: {content}"}
            
            for content in response:
                last_content.append(format_content(content))

    async def process_transcription(self) -> None:
        from ..bridge import bridge

        if (last_audios := self.last_message_audios()):
            audio_obj = last_audios[0]

            if audio_obj.get("type") == "input_audio":
                b64_str = audio_obj.get("input_audio", {}).get("data", "")
            else:
                b64_str = audio_obj.get("audio_url", {}).get("url", "").split(",")[-1]

            audio_bytes = base64.b64decode(b64_str)

            transcription_payload = {
                "audio": audio_bytes,
            }

            headers = {}

            response: str = await bridge.process_stt(transcription_payload, headers)
            
            # Garante que o conteúdo seja uma lista
            current_content = self.messages[-1].get("content", "")
            if isinstance(current_content, str):
                self.messages[-1]["content"] = [{"type": "text", "text": current_content}]
            
            last_content: list = self.messages[-1]["content"]
            last_content.append({"type": "text", "text": f"[User sent audio]: {response}"})
            
            # Remove os áudios originais
            self.messages[-1]["content"] = [part for part in self.messages[-1]["content"] if part.get("type") not in ["audio_url", "input_audio"]]
