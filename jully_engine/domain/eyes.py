import logging
import base64
import io
from PIL import Image
from typing import Any, Dict, List, Optional

from ..services.vision import FaceDetector, FaceService

from ..engine_models.tagger import ONNXTagger
from ..engine_models.gguf import GGUF
from ..engine_models.fastvlm import FastVLM
from ..engine_models.emotion import Emotion
from ..engine_models.llm_api import LLMApi
from ..engine_models.moondream import MoondreamVLM

logger = logging.getLogger("JulyEngine.Domain.Eyes")

class Eyes:
    """
    Handles vision and image analysis.
    Strategies: GGUF (cpu, gpu), FastVLM (cpu, gpu), MoondreamVLM (cpu, gpu), Emotion (cpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()
        self.face_service = FaceService()

    def _get_strategy(self):
        if self.backend == "api":
            return LLMApi(backend=self.backend)
        elif self.model_tag == "emotion":
            return Emotion(FaceDetector(), backend="cpu")
        elif self.model_tag == "fastvlm":
            return FastVLM(backend=self.backend)
        elif self.model_tag == "moondream":
            return MoondreamVLM(backend=self.backend)
        elif self.model_tag == "tagger" and self.backend == 'cpu':
            return ONNXTagger()
        
        from ..persistence.persistence import get_backend
        model = get_backend().get_model(self.model_tag)

        if self.backend in ["gpu", "cpu"]:
            return GGUF(backend=self.backend, model=model)
        else:
            raise ValueError(f"Eyes: Unsupported backend/model combination: {self.backend}/{self.model_tag}")
    
    def decode_image(self, image_data):
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            
            img_bytes = base64.b64decode(image_data)
            
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return Image.open(io.BytesIO(image_data)).convert("RGB")

    async def analyze(self, payload: Dict[str, Any]):
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

        if isinstance(self._strategy, LLMApi):
            model = payload.pop("model", self.model_tag)
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            headers = payload.pop("headers", {})
            return await self._strategy.run_chat(model, messages, stream=stream, headers=headers, **payload)

        # Generalize extraction for local models
        images_data = payload.get("images", [])
        if not images_data and payload.get("image"):
            images_data = [payload.get("image")]
            
        prompt = payload.get("prompt", "")
        
        if not images_data:
            messages = payload.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.get("content", [])
            
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                images_data.append(url.split(",")[1])
                            else:
                                images_data.append(url)
                        elif part.get("type") == "text":
                            prompt = part.get("text", prompt)
            
            if images_data:
                payload["images"] = images_data
            if prompt:
                payload["prompt"] = prompt

        if isinstance(self._strategy, Emotion):
            if not images_data:
                raise ValueError("Eyes: No image data for emotion analysis")
            
            results = []
            for img_data in images_data:
                img = self.decode_image(img_data)
                results.append(self._strategy.run(img))
            
            return results if len(images_data) > 1 else results[0]

        elif isinstance(self._strategy, FastVLM):
            if len(images_data) > 1:
                return self._strategy.run_batch(images_data, prompt)
            return self._strategy.run(payload)

        elif isinstance(self._strategy, MoondreamVLM):
            if len(images_data) > 1:
                return self._strategy.run_batch(images_data, prompt)
            return self._strategy.run(payload)

        elif isinstance(self._strategy, GGUF):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False)
            # GGUF doesn't support batching natively in this implementation yet, 
            # so we run it one by one if multiple images provided
            if len(images_data) > 1:
                results = []
                for img_data in images_data:
                    single_msg = [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}" if not img_data.startswith("http") else img_data}},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                    results.append(await self._strategy.run_chat(single_msg, stream=False))
                return results
            
            return self._strategy.run_chat(messages, stream=stream, **payload)

        elif isinstance(self._strategy, ONNXTagger):
            if not images_data:
                raise ValueError("Eyes: No image data for tagger analysis")
            
            results = [self._strategy.tag(self.decode_image(img_data)) for img_data in images_data]
            return results if len(images_data) > 1 else results[0]

    def describe_person_faces(self, image: Image.Image) -> str:
        """Decoupled logic to describe faces using VLM without saving temp files."""
        descriptions = []
        
        # O "Prompt Bisturi" restritivo para forçar descrições curtas
        strict_prompt = (
            "Act as a strict facial feature extractor. Describe the person in the image in a single, short sentence. "
            "Focus ONLY on: gender, hair color/style, eye color, and visible accessories. "
            "CRITICAL: Do NOT guess emotions. Do NOT describe the background. "
            "Example: Man with brown hair and green eyes, wearing round gold glasses."
        )

        # Consumimos o yield duplo (Vetor Matemático, Matriz da Imagem)
        for emb, face_crop in self.face_service.get_faces_embeddings(image):
            try:
                # 1. Converte a matriz Numpy (RGB) de volta para Base64 em memória
                pil_crop = Image.fromarray(face_crop)
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                description = ""

                # 2. Roteamento Inteligente baseado na Strategy atual
                if isinstance(self._strategy, FastVLM):
                    payload = {"image": img_b64, "prompt": strict_prompt}
                    # Executa a inferência síncrona
                    description = self._strategy.run(payload)

                elif isinstance(self._strategy, GGUF):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": strict_prompt}
                        ]
                    }]
                    # Executa a inferência síncrona
                    description = self._strategy.run_chat(messages, stream=False)
                
                else:
                    logger.warning(f"Strategy {type(self._strategy)} não suportada para describe_person_faces síncrono.")

                # 3. Limpeza do resultado
                if description:
                    # Remove quebras de linha que o LLM possa ter alucinado
                    clean_desc = description.replace('\n', ' ').strip()
                    descriptions.append(clean_desc)

            except Exception as e:
                logger.error(f"Erro ao gerar descrição do rosto com VLM: {e}")

        # Se houver múltiplos rostos na foto, concatenamos com um separador elegante
        return " | ".join(descriptions) if descriptions else ""

    def unload(self):
        """Libera os recursos da estratégia (VLM, GGUF, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload(self.model_tag)
            logger.info(f"Eyes: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Eyes: Strategy {self.model_tag} cleared.")