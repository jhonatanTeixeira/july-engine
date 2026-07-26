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

    def get_videos(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return [
            part for part in self._get_multimodal_content(messages)
            if part.get("type", "text") == "video_url"
        ]

    def get_files(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return [
            part for part in self._get_multimodal_content(messages)
            if part.get("type", "text") == "file_url"
        ]

    def _filter_content_by_type(self, types_to_remove: List[str], messages: Optional[List[Dict]] = None) -> List[Dict]:
        msgs = deepcopy(messages if messages is not None else self.messages)
        if not msgs: return []

        for msg in msgs:
            content = msg.get("content", "")

            if isinstance(content, list):
                filtered = [part for part in content if part.get("type") not in types_to_remove]
                # Normalize: plain string when only one text part remains, empty string
                # when nothing is left — avoids sending bare lists to the LLM API.
                if not filtered:
                    msg["content"] = ""
                elif len(filtered) == 1 and filtered[0].get("type") == "text":
                    msg["content"] = filtered[0].get("text", "")
                else:
                    msg["content"] = filtered

        return msgs

    def filter_images(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["image_url"], messages)

    def filter_audios(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["audio_url", "input_audio"], messages)

    def filter_videos(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["video_url"], messages)

    def filter_pdfs(self, messages: Optional[List[Dict]] = None) -> List[Dict]:
        return self._filter_content_by_type(["file_url"], messages)

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

    def last_message_videos(self) -> List[Dict]:
        if not self.messages: return []
        return self.get_videos([self.messages[-1]])

    def last_message_files(self) -> List[Dict]:
        if not self.messages: return []
        return self.get_files([self.messages[-1]])

    def _append_text_part(self, text: str) -> None:
        current_content = self.messages[-1].get("content", "")
        if isinstance(current_content, str):
            self.messages[-1]["content"] = [{"type": "text", "text": current_content}]
        self.messages[-1]["content"].append({"type": "text", "text": text})

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

    def sanitize_images(self, max_size: int = 1024):
        """Redimensiona e converte todas as imagens no payload para JPEG base64 limpo."""
        import io
        try:
            from PIL import Image
        except ImportError:
            return

        for message in self.messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            
            for part in content:
                if part.get("type") == "image_url":
                    img_url_obj = part.get("image_url", {})
                    url = img_url_obj.get("url", "")
                    
                    if url.startswith("data:image"):
                        try:
                            # Extrai o base64
                            header, b64_data = url.split(",", 1)
                            img_bytes = base64.b64decode(b64_data)
                            
                            # Processa com Pillow
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            # Se tiver transparência, compõe em fundo branco
                            if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
                                if img.mode != "RGBA":
                                    img = img.convert("RGBA")
                                background = Image.new("RGBA", img.size, (255, 255, 255))
                                img = Image.alpha_composite(background, img).convert("RGB")
                            elif img.mode != "RGB":
                                img = img.convert("RGB")
                                
                            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                            
                            # Salva como JPEG
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=85)
                            new_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                            
                            # Atualiza a URL
                            img_url_obj["url"] = f"data:image/jpeg;base64,{new_b64}"
                        except Exception as e:
                            import logging
                            logging.getLogger("JulyEngine.Helpers").warning(f"Failed to sanitize image: {e}")

    def sanitize_audio(self) -> None:
        """No-op placeholder for the native-audio branch: audio_url/input_audio parts
        are left as-is (MTMD decodes them itself), mirroring sanitize_images()'s role
        for vision — kept as a method so chat_adapter.run() has one call to make
        regardless of which branch (native vs. transcribe-fallback) is active."""
        pass

    async def process_video(self) -> None:
        """Always runs — no local model here understands raw video natively. Decodes
        the last message's video, runs it through the existing video-description
        pipeline (frame sampling + VLM captioning, same one POST /vision/video/describe
        already uses), and splices the result in as text."""
        from ..bridge import bridge

        if not (last_videos := self.last_message_videos()):
            return

        video_obj = last_videos[0]
        b64_str = video_obj.get("video_url", {}).get("url", "").split(",")[-1]
        video_bytes = base64.b64decode(b64_str)

        import os
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            # Same defaults POST /vision/video/describe's Form fields use.
            aggregate = await bridge.process_video_description({
                "video_path": tmp_path,
                "interval_sec": 2.0,
                "frames_per_grid": 4,
                "strategy": "default",
                "detect_changes": False,
            }, {})

            parts = []
            if getattr(aggregate, "full_transcription", None):
                parts.append(aggregate.full_transcription)
            for segment in getattr(aggregate, "segments", None) or []:
                if segment.narrative and segment.narrative.text:
                    parts.append(segment.narrative.text)
            description = "\n".join(parts) if parts else "sem descrição disponível"
        except Exception as e:
            import logging
            logging.getLogger("JulyEngine.Helpers").warning(f"Failed to process video: {e}")
            description = "falha ao processar vídeo"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        self._append_text_part(f"[Vídeo enviado pelo usuário]: {description}")

    async def process_pdf(self) -> None:
        """Always runs — no local model here understands raw PDF natively. Extracts
        text via the existing PDF pipeline (same one POST /utils/extract-pdf already
        uses) and splices it in as text."""
        from ..bridge import bridge

        if not (last_files := self.last_message_files()):
            return

        file_obj = last_files[0]
        b64_str = file_obj.get("file_url", {}).get("url", "").split(",")[-1]
        pdf_bytes = base64.b64decode(b64_str)

        try:
            events = await bridge.process_pdf_extract(pdf_bytes)
            # extract_pdf() is a synchronous, CPU-bound generator (PyMuPDF/pytesseract) —
            # consume it in a thread so it doesn't block the event loop.
            text_parts = await asyncio.to_thread(
                lambda: [e.get("text", "") for e in events if e.get("type") == "text"]
            )
            joined = "\n".join(p for p in text_parts if p)
            description = joined if joined else "sem texto extraído"
        except Exception as e:
            import logging
            logging.getLogger("JulyEngine.Helpers").warning(f"Failed to process PDF: {e}")
            description = "falha ao processar PDF"

        self._append_text_part(f"[PDF enviado pelo usuário]: {description}")
