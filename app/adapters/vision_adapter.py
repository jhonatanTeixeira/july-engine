import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from ..models.base_model import BaseModel

logger = logging.getLogger("JulyEngine.Adapter.VisionAdapter")

_TASK_HANDLERS = {
    "vision_chat":       "_analyze",
    "video_description": "_describe_video",
    "face_extraction":   "_describe_person_faces",
    "face_sync":         "_sync_faces_batch",
}


class VisionAdapter(BaseModel):
    """
    Handles all vision task types: vision_chat, video_description,
    face_extraction, face_sync.

    Engine field: "vision"

    Sub-engine selection:
      meta["vision_engine"] → explicit engine name
      alias prefix matching → fastvlm | moondream | emotion | tagger
      backend == "api"      → LLMApi route (handled inline)
    """

    def __init__(self, task_type: str, backend: str = "gpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.task_type = task_type
        self._vision_model = None
        self._face_service = None

    # ------------------------------------------------------------------
    # Sub-engine resolution
    # ------------------------------------------------------------------

    def _detect_engine(self) -> str:
        if self.backend == "api":
            return "api"
        
        return (self.meta.get("alias") or self.meta.get("model", "")).lower()

    def _get_vision_model(self):
        engine = self._detect_engine()

        if engine == "fastvlm":
            from ..models.fastvlm import FastVLMModel
            self._vision_model = FastVLMModel(backend=self.backend, model_meta=self.meta)
        elif engine == "moondream":
            from ..models.moondream import MoondreamModel
            self._vision_model = MoondreamModel(backend=self.backend, model_meta=self.meta)
        elif engine == "emotion":
            from ..models.emotion import EmotionModel
            from ..services.vision import FaceDetector
            self._vision_model = EmotionModel(FaceDetector(), backend=self.backend)
        elif engine == "tagger":
            from ..models.tagger import TaggerModel
            self._vision_model = TaggerModel()
        else:
            raise ValueError(f"no model found for metadata {self.meta}")

        return self._vision_model

    def _get_face_service(self):
        if self._face_service is None:
            from ..services.vision import FaceService
            self._face_service = FaceService()
        return self._face_service

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        m = self._get_vision_model()
        if m and hasattr(m, "get_required_vram"):
            result = m.get_required_vram(payload)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return 0

    def load(self, n_ctx=None, num_layers=None):
        m = self._get_vision_model()
        if m:
            m.load()

    def is_loaded(self) -> bool:
        return self._vision_model is not None and self._vision_model.is_loaded()

    def unload(self, model_name=None):
        if self._vision_model:
            self._vision_model.unload(model_name or self.model_id)

    async def run(self, payload: Dict[str, Any]):
        task_type = self.task_type
        handler_name = _TASK_HANDLERS.get(task_type)

        if not handler_name:
            raise ValueError(f"VisionAdapter: unknown task_type '{task_type}'")
        
        return await getattr(self, handler_name)(payload)

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------

    @staticmethod
    def decode_image(image_data) -> "Image":
        from PIL import Image

        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        
        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]
            elif image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[-1]
        
            image_data = re.sub(r'[^a-zA-Z0-9+/=]', '', image_data)
            missing = len(image_data) % 4
        
            if missing:
                image_data += "=" * (4 - missing)
        
            img_bytes = base64.b64decode(image_data)
            from PIL import Image
        
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        from PIL import Image
        
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    @staticmethod
    def _sanitize_image(b64_str: str, max_size: int = 1024) -> str:
        from PIL import Image
        try:
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"VisionAdapter: image sanitize failed, using original: {e}")
            return b64_str

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
        return str(response)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _analyze(self, payload: Dict[str, Any]) -> List[str]:
        engine = self._detect_engine()
        headers = payload.get("headers", {})

        # Inject config headers from TEXT_PRESETS when using API or GGUF
        if engine in ("api", "gguf"):
            config = self._load_preset_config()
            if config:
                headers.setdefault("x-base-url", config.get("base_url"))
                if config.get("api_key"):
                    headers.setdefault("x-api-key", config["api_key"])
                    headers.setdefault("authorization", f"Bearer {config['api_key']}")

        if engine in ("api", "gguf"):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)

            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:image"):
                                hdr, raw_b64 = url.split(",", 1)
                                small = self._sanitize_image(raw_b64, max_size=1024)
                                part["image_url"]["url"] = f"data:image/jpeg;base64,{small}"

            if engine == "api":
                from ..services.llm_api import llm_api
                model_tag = payload.pop("model", self.meta.get("alias", ""))
                raw = await llm_api.dispatch("vision_chat", {
                    **payload, "messages": messages, "stream": stream,
                    "model": model_tag, "headers": headers,
                })
            else:
                model = self._get_vision_model()
                raw = await model.run_chat(messages, stream=stream, **payload)

            return [self._extract_text(raw)]

        # Local VLM / Emotion / Tagger — extract images from messages
        extracted_images, extracted_prompt = self._extract_images_from_payload(payload)
        if not extracted_images:
            raise ValueError("VisionAdapter: no image data found in payload")

        if engine == "emotion":
            results = []
            model = self._get_vision_model()
            for b64 in extracted_images:
                img = self.decode_image(b64)
                raw = model.run({"image": img})
                results.append(json.dumps(raw) if isinstance(raw, dict) else str(raw))
            return results

        if engine == "tagger":
            results = []
            model = self._get_vision_model()
            for b64 in extracted_images:
                img = self.decode_image(b64)
                tags = model.run({"image": img})
                results.append(", ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags))
            return results

        # fastvlm / moondream
        model = self._get_vision_model()
        if len(extracted_images) > 1 and hasattr(model, "run_batch"):
            batch = model.run_batch(extracted_images, extracted_prompt)
            return [str(r) for r in batch] if isinstance(batch, list) else [str(batch)]

        result = model.run({"image": extracted_images[0], "prompt": extracted_prompt})
        return [str(result)]

    def _extract_images_from_payload(self, payload: dict):
        images, prompt = [], ""
        for msg in payload.get("messages", []):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    raw = url.split(",", 1)[1] if url.startswith("data:") else url
                    images.append(self._sanitize_image(raw, max_size=768))
                elif part.get("type") == "text":
                    prompt = part.get("text", "")
        return images, prompt

    async def _describe_person_faces(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        images_b64 = []
        for msg in payload.get("messages", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        images_b64.append(url.split(",", 1)[1] if url.startswith("data:") else url)
            elif isinstance(content, str) and content.startswith("data:image"):
                images_b64.append(content.split(",", 1)[1])

        if not images_b64 and payload.get("images"):
            images_b64 = payload["images"]

        images_pil = [self.decode_image(b) for b in images_b64]
        collection = payload.get("collection", "faces_embeddings")
        face_service = self._get_face_service()

        strict_prompt = (
            "Act as a strict facial feature extractor. Describe the person in the image in a single, short sentence. "
            "Focus ONLY on: gender, hair color/style, eye color, and visible accessories. "
            "CRITICAL: Do NOT guess emotions. Do NOT describe the background. "
            "Example: Man with brown hair and green eyes, wearing round gold glasses.\n"
            "Please provide a description for each face in order, separated by a newline."
        )

        batch_crops: List[str] = []
        batch_embeddings = []

        for idx, img in enumerate(images_pil):
            faces = list(face_service.get_faces_embeddings(img))
            for emb, face_crop, _ in faces:
                try:
                    from PIL import Image as PILImage
                    pil_crop = PILImage.fromarray(face_crop)
                    buf = io.BytesIO()
                    pil_crop.save(buf, format="JPEG")
                    batch_crops.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                    batch_embeddings.append(emb)
                except Exception as e:
                    logger.error(f"VisionAdapter: face crop encode error: {e}")

        if not batch_crops:
            return []

        descriptions = []
        try:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                for b in batch_crops
            ]
            content.append({"type": "text", "text": strict_prompt})
            from ..bridge import bridge
            res = await bridge.process_image_description(
                {"messages": [{"role": "user", "content": content}], "model": self.meta.get("alias", "")},
                payload.get("headers", {}),
            )
            text = self._extract_text(res)
            descriptions = [d.strip() for d in text.split("\n") if d.strip()]
        except Exception as e:
            logger.error(f"VisionAdapter: face description LLM call failed: {e}")

        while len(descriptions) < len(batch_embeddings):
            descriptions.append("Unknown description")

        results = []
        for emb, desc in zip(batch_embeddings, descriptions):
            person_id, emb = face_service.match_or_add_face(emb, pic_id="api_request", collection=collection)
            results.append({"person_id": person_id, "embedding": emb, "description": desc})

        return results

    async def _sync_faces_batch(self, payload: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        images_b64 = payload.get("images", [])
        pic_ids = payload.get("pic_ids", [])
        collection = payload.get("collection", "faces_embeddings")
        images_pil = [self.decode_image(b) for b in images_b64]
        face_service = self._get_face_service()
        return face_service.sync_faces_batch(images_pil, pic_ids, collection)

    async def _describe_video(self, payload: Dict[str, Any]) -> Any:
        import os
        from ..services.video_processing import multimodal_video_analysis

        video_path = payload.get("video_path")
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"VisionAdapter: video path invalid or missing: {video_path}")

        aggregate = await multimodal_video_analysis.execute(
            video_path=video_path,
            interval_sec=float(payload.get("interval_sec", 2.0)),
            frames_per_grid=int(payload.get("frames_per_grid", 4)),
            strategy=payload.get("strategy", "default"),
            detect_changes=payload.get("detect_changes", "false") == "true",
            headers=payload.get("headers", {}),
        )

        description_model = payload.get("description_model")
        if not description_model:
            return aggregate

        # Build synthesis prompt
        prompt_parts = [
            "You are an expert video analyst. I will provide you with a chronological breakdown "
            "of a video's visual segments and its full audio transcription.",
            "Please synthesize this information into a single, cohesive, highly detailed and fluid narrative.",
            "Combine the visual actions with the spoken words to give full context.",
            "\n=== AUDIO TRANSCRIPTION ===",
            aggregate.full_transcription if aggregate.full_transcription else "[No speech detected]",
            "\n=== VISUAL TIMELINE ===",
        ]
        for seg in aggregate.segments:
            desc = seg.narrative.text if hasattr(seg.narrative, "text") else str(seg.narrative)
            prompt_parts.append(f"[{seg.start_offset:.1f}s to {seg.end_offset:.1f}s]: {desc}")

        from ..bridge import bridge
        result = await bridge.process_openai_chat(
            {
                "model": description_model,
                "messages": [
                    {"role": "system", "content": "You are a multimodal video synthesis AI."},
                    {"role": "user", "content": "\n".join(prompt_parts)},
                ],
                "stream": False,
                "headers": payload.get("headers", {}),
            },
            payload.get("headers", {}),
        )

        if isinstance(result, dict) and "choices" in result:
            return result["choices"][0]["message"].get("content", "")
        return str(result)

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _load_preset_config(self) -> dict:
        try:
            from ..services.models_service import model_service
            presets = model_service.backend.get_setting("TEXT_PRESETS") or []
            alias = self.meta.get("alias", "")
            config = next((p for p in presets if p.get("alias") == alias), None)
            return config or (presets[0] if presets else {})
        except Exception:
            return {}
