from __future__ import annotations
import asyncio
import logging
import base64
import io
import json
import inspect
from PIL import Image
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_models.tagger import ONNXTagger
    from ..engine_models.llama_gguf import GGUF
    from ..engine_models.fastvlm import FastVLM
    from ..engine_models.emotion import Emotion
    from ..engine_models.llm_api import LLMApi
    from ..engine_models.moondream import MoondreamVLM

from ..services.vision import FaceDetector, FaceService

logger = logging.getLogger("JulyEngine.Domain.Eyes")

class Eyes:
    """
    Handles vision and image analysis.
    Strategies: GGUF (cpu, gpu), FastVLM (cpu, gpu), MoondreamVLM (cpu, gpu), Emotion (cpu), LLMApi (api).
    Contract: analyze() ALWAYS returns a List[str].
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()
        self.face_service = FaceService()

    def _get_strategy(self):
        logger.debug(f"Eyes: Resolving strategy for backend='{self.backend}', model_tag='{self.model_tag}'")
        if self.backend == "api":
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)
        elif self.model_tag == "emotion":
            from ..engine_models.emotion import Emotion
            return Emotion(FaceDetector(), backend="cpu")
        elif self.model_tag == "fastvlm":
            from ..engine_models.fastvlm import FastVLM
            return FastVLM(backend=self.backend)
        elif self.model_tag == "moondream":
            from ..engine_models.moondream import MoondreamVLM
            return MoondreamVLM(backend=self.backend)
        elif self.model_tag == "tagger" and self.backend == 'cpu':
            from ..engine_models.tagger import ONNXTagger
            return ONNXTagger()
        
        from ..persistence.persistence import get_backend
        model = get_backend().get_model(self.model_tag)

        if self.backend in ["gpu", "cpu"]:
            from ..engine_models.llama_gguf import GGUF
            strategy = GGUF(backend=self.backend, model=model)
            logger.info(f"Eyes: Using GGUF strategy for {self.model_tag}")
            return strategy
        else:
            logger.error(f"Eyes: Unsupported backend/model combination: {self.backend}/{self.model_tag}")
            raise ValueError(f"Eyes: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            res = self._strategy.get_required_vram(payload)
            if inspect.iscoroutine(res):
                return await res
            return res
        return 0
    
    def decode_image(self, image_data: str) -> Image.Image:
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def _extract_text(self, response: Any) -> str:
        """Helper to force OpenAI-like responses into pure text."""
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
        return str(response)

    def _sanitize_image(self, base64_str: str, max_size: int = 1024) -> str:
        """
        Decodifica, redimensiona (downscale) mantendo a proporção, 
        comprime para JPEG e devolve em Base64.
        """
        import base64
        import io
        from PIL import Image

        try:
            # 1. Decodifica o base64
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            orig_size = img.size
            
            # Remove o canal Alpha (Transparência) se for PNG, pois VLMs preferem RGB puro
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # 2. Downscale super rápido e seguro
            # Se a imagem for menor que max_size, o thumbnail não faz nada!
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            new_size = img.size
            
            # 3. Re-encoda para JPEG com compressão leve
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            
            logger.debug(f"Eyes: Image sanitized from {orig_size} to {new_size}")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            # Se falhar (ex: string malformada), devolve o original e reza para a Engine aguentar
            logger.warning(f"Falha ao sanitizar imagem. Usando original: {e}")
            return base64_str

    async def analyze(self, payload: Dict[str, Any]) -> List[str]:
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.llama_gguf import GGUF
        from ..engine_models.emotion import Emotion
        from ..engine_models.tagger import ONNXTagger
        from ..engine_models.fastvlm import FastVLM
        from ..engine_models.moondream import MoondreamVLM

        logger.info(f"Eyes: Starting analysis with strategy {type(self._strategy).__name__}")
        headers = payload.get("headers", {})
        
        from ..persistence import get_backend
        backend_db = get_backend()
        text_presets = backend_db.get_setting("TEXT_PRESETS") or []
        config = next((p for p in text_presets if p.get("alias") == self.model_tag), None)
        
        if not config and text_presets:
            config = text_presets[0]
            
        if config:
            if "x-base-url" not in headers and "base_url" in config:
                headers["x-base-url"] = config["base_url"]
            has_auth = "authorization" in headers or "x-api-key" in headers
            if not has_auth and "api_key" in config and config["api_key"]:
                headers["x-api-key"] = config["api_key"]
                headers["authorization"] = f"Bearer {config['api_key']}"

        # 1. Rotas Nativas OpenAI (LLMApi e GGUF)
        if isinstance(self._strategy, (LLMApi, GGUF)):
            logger.debug("Eyes: Using OpenAI-compatible route (LLMApi/GGUF)")
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:image"):
                                header, raw_b64 = url.split(",", 1)
                                small_b64 = self._sanitize_image(raw_b64, max_size=1024)
                                part["image_url"]["url"] = f"data:image/jpeg;base64,{small_b64}"
            
            if isinstance(self._strategy, LLMApi):
                model = payload.pop("model", self.model_tag)
                headers = payload.pop("headers", headers)
                raw_response = await self._strategy.run_chat(model, messages, stream=stream, headers=headers, **payload)
            else:
                raw_response = await self._strategy.run_chat(messages, stream=stream, **payload)
                
            logger.info("Eyes: OpenAI-compatible analysis completed.")
            return [self._extract_text(raw_response)]

        # =====================================================================
        # EXTRAÇÃO SEGURA DE DADOS CRUS (Para FastVLM, Moondream, ONNX, Emotion)
        # =====================================================================
        extracted_images = []
        extracted_prompt = ""
        
        messages = payload.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", [])
        
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        raw_b64 = url.split(",", 1)[1] if url.startswith("data:") else url
                        
                        safe_b64 = self._sanitize_image(raw_b64, max_size=768)
                        extracted_images.append(safe_b64)
                        
                    elif part.get("type") == "text":
                        extracted_prompt = part.get("text", "")

        if not extracted_images:
            raise ValueError(f"Eyes: No image data found in payload for strategy {self.model_tag}")

        # 2. Rotas de Modelos Específicos (Garantindo Retorno List[str])
        
        if isinstance(self._strategy, Emotion):
            logger.info(f"Eyes: Running emotion analysis on {len(extracted_images)} images.")
            results = []
            for img_data in extracted_images:
                img = self.decode_image(img_data)
                raw_res = self._strategy.run(img)
                text_res = json.dumps(raw_res) if isinstance(raw_res, dict) else str(raw_res)
                results.append(text_res)
            return results

        elif isinstance(self._strategy, ONNXTagger):
            logger.info(f"Eyes: Running ONNX Tagger on {len(extracted_images)} images.")
            results = []
            for img_data in extracted_images:
                tags = self._strategy.tag(self.decode_image(img_data))
                if isinstance(tags, list):
                    results.append(", ".join(str(t) for t in tags))
                else:
                    results.append(str(tags))
            return results

        elif isinstance(self._strategy, (FastVLM, MoondreamVLM)):
            logger.info(f"Eyes: Running Local VLM ({type(self._strategy).__name__})")
            if len(extracted_images) > 1:
                batch_result = self._strategy.run_batch(extracted_images, extracted_prompt)
                if isinstance(batch_result, list):
                    return [str(r) for r in batch_result]
                return [str(batch_result)]
            
            single_payload = {"image": extracted_images[0], "prompt": extracted_prompt}
            single_result = self._strategy.run(single_payload)
            logger.info("Eyes: Local VLM analysis completed.")
            return [str(single_result)]

    async def describe_person_faces(self, images: Union[Image.Image, List[Image.Image]], collection: str = "faces_embeddings") -> List[Dict[str, Any]]:
        """High-Performance batched face orchestrator."""
        if not isinstance(images, list):
            images = [images]

        logger.info(f"Eyes: Extracting faces from {len(images)} images.")
        strict_prompt = (
            "Act as a strict facial feature extractor. Describe the person in the image in a single, short sentence. "
            "Focus ONLY on: gender, hair color/style, eye color, and visible accessories. "
            "CRITICAL: Do NOT guess emotions. Do NOT describe the background. "
            "Example: Man with brown hair and green eyes, wearing round gold glasses."
            "\nPlease provide a description for each face in order, separated by a newline."
        )

        batch_crops = []
        batch_embeddings = []

        for idx, img in enumerate(images):
            faces = list(self.face_service.get_faces_embeddings(img))
            logger.debug(f"Eyes: Image {idx} contains {len(faces)} faces.")
            for emb, face_crop, bbox in faces:
                try:
                    pil_crop = Image.fromarray(face_crop)
                    buffered = io.BytesIO()
                    pil_crop.save(buffered, format="JPEG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    batch_crops.append(img_b64)
                    batch_embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Erro ao converter crop de rosto: {e}")

        if not batch_crops:
            logger.warning("Eyes: No faces detected in provided images.")
            return []

        logger.info(f"Eyes: Total faces to describe: {len(batch_crops)}")

        descriptions = []
        from ..services.helpers import inference_helper
        
        try:
            messages = [{"role": "user", "content": []}]
            for b64 in batch_crops:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            messages[0]["content"].append({"type": "text", "text": strict_prompt})
            
            res = await inference_helper.process('vision_chat', {"messages": messages, "model": self.model_tag})
            text_response = self._extract_text(res)
            descriptions = [d.strip() for d in text_response.split('\n') if d.strip()]
            
            while len(descriptions) < len(batch_embeddings):
                descriptions.append("Unknown description")
                
        except Exception as e:
            logger.error(f"Erro ao gerar descrição de rosto em lote com VLM: {e}")
            descriptions = ["Unknown"] * len(batch_embeddings)

        results = []
        for emb, desc in zip(batch_embeddings, descriptions):
            # Delega para o FaceService a lógica de match, EMA e inserção
            person_id, emb = self.face_service.match_or_add_face(emb, pic_id="api_request", collection=collection)
            logger.debug(f"Eyes: Matched/Added person_id='{person_id}'")

            results.append({
                "person_id": person_id,
                "embedding": emb,
                "description": desc
            })
            
        logger.info(f"Eyes: Face extraction completed. Found {len(results)} people.")
        return results

    async def sync_faces_batch(self, images: List[Image.Image], pic_ids: List[str], collection: str = "faces_embeddings") -> List[List[Dict[str, Any]]]:
        return self.face_service.sync_faces_batch(images, pic_ids, collection)

    async def describe_video(self, payload: Dict):
        """
        Analisa um vídeo e retorna uma descrição (str) ou o VideoAggregate bruto.

        - Se `description_model` estiver presente no payload, sintetiza o aggregate em uma
          narrativa textual usando o LLM indicado e retorna uma str.
        - Se `description_model` ausente, retorna o VideoAggregate bruto para que o
          chamador possa processar a estrutura diretamente via API.
        """
        import os
        from ..services.video_processing import multimodal_video_analysis
        from ..services.helpers import inference_helper

        video_path = payload.get("video_path")
        interval_sec = float(payload.get("interval_sec", 2.0))
        frames_per_grid = int(payload.get("frames_per_grid", 4))
        strategy = payload.get("strategy", "default")
        detect_changes = payload.get("detect_changes", "false") == "true"

        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Eyes: Video path is invalid or missing: {video_path}")

        # 1. Roda a análise multimodal pesada (Grids Visuais + STT em paralelo)
        logger.info(f"Eyes.describe_video: processing video {video_path} (interval={interval_sec}s, grid={frames_per_grid})")
        aggregate = await multimodal_video_analysis.execute(
            video_path=video_path,
            interval_sec=interval_sec,
            frames_per_grid=frames_per_grid,
            strategy=strategy,
            detect_changes=detect_changes
        )

        # 2. Se não houver description_model, retorna o aggregate bruto para a API
        description_model = payload.get("description_model")
        if not description_model:
            logger.info("Eyes.describe_video: no description_model provided — returning raw VideoAggregate.")
            return aggregate

        # 3. Monta o Dossiê (Super-Prompt) combinando Visão e Áudio
        prompt_parts = [
            "You are an expert video analyst. I will provide you with a chronological breakdown of a video's visual segments and its full audio transcription.",
            "Please synthesize this information into a single, cohesive, highly detailed and fluid narrative of what happens in the video.",
            "Combine the visual actions with the spoken words to give full context.",
            "\n=== AUDIO TRANSCRIPTION ===",
            aggregate.full_transcription if aggregate.full_transcription else "[No speech detected]",
            "\n=== VISUAL TIMELINE ==="
        ]

        # Costura a linha do tempo
        for seg in aggregate.segments:
            desc = seg.narrative.text if hasattr(seg.narrative, 'text') else str(seg.narrative)
            prompt_parts.append(f"[{seg.start_offset:.1f}s to {seg.end_offset:.1f}s]: {desc}")

        final_prompt = "\n".join(prompt_parts)

        # 4. Repassa para a Inteligência de Texto (O Roteador/Brain)
        llm_payload = {
            "messages": [
                {"role": "system", "content": "You are a multimodal video synthesis AI."},
                {"role": "user", "content": final_prompt}
            ],
            "headers": {"x-context-window": payload.get("headers", {}).get("x-context-window", None)},
            "model": description_model,
            "stream": False
        }

        logger.info(f"Eyes.describe_video: synthesizing with description_model='{description_model}'")
        synthesis_result = await inference_helper.process("text_chat", llm_payload)

        # 5. Extrai a string pura do retorno padrão da OpenAI
        if isinstance(synthesis_result, dict) and "choices" in synthesis_result:
            logger.info("Eyes.describe_video: synthesis completed successfully.")
            return synthesis_result["choices"][0]["message"].get("content", "")

        logger.info("Eyes.describe_video: synthesis completed (raw format).")
        return str(synthesis_result)

    def is_loaded(self):
        return hasattr(self._strategy, "is_loaded") and self._strategy.is_loaded()

    def load(self):
        if hasattr(self._strategy, "load"):
            logger.info(f"Eyes: Loading state for model {self.model_tag}")
            self._strategy.load()

    def unload(self):
        """Libera os recursos da estratégia (VLM, GGUF, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload(self.model_tag)
            logger.info(f"Eyes: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Eyes: Strategy {self.model_tag} cleared.")