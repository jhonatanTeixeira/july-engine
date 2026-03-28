import logging
import base64
import io
import json
from PIL import Image
from typing import Any, Dict, List, Optional, Union

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
    Contract: analyze() ALWAYS returns a List[str].
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

    async def analyze(self, payload: Dict[str, Any]) -> List[str]:
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
        # Ambas recebem o payload limpo e retornam [String]
        if isinstance(self._strategy, (LLMApi, GGUF)):
            messages = payload.pop("messages", [])
            stream = payload.pop("stream", False) # Forçamos False para garantir o contrato List[str]
            
            if isinstance(self._strategy, LLMApi):
                model = payload.pop("model", self.model_tag)
                headers = payload.pop("headers", headers)
                raw_response = await self._strategy.run_chat(model, messages, stream=stream, headers=headers, **payload)
            else:
                raw_response = await self._strategy.run_chat(messages, stream=stream, **payload)
                
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
                        extracted_images.append(url.split(",")[1] if url.startswith("data:") else url)
                    elif part.get("type") == "text":
                        extracted_prompt = part.get("text", "")

        if not extracted_images:
            raise ValueError(f"Eyes: No image data found in payload for strategy {self.model_tag}")

        # 2. Rotas de Modelos Específicos (Garantindo Retorno List[str])
        
        if isinstance(self._strategy, Emotion):
            results = []
            for img_data in extracted_images:
                img = self.decode_image(img_data)
                # Garante que o retorno do Emotion vira texto (mesmo que seja dict nativamente)
                raw_res = self._strategy.run(img)
                text_res = json.dumps(raw_res) if isinstance(raw_res, dict) else str(raw_res)
                results.append(text_res)
            return results

        elif isinstance(self._strategy, ONNXTagger):
            results = []
            for img_data in extracted_images:
                tags = self._strategy.tag(self.decode_image(img_data))
                # Tagger retorna lista de tags. Juntamos tudo em uma string separada por vírgula.
                if isinstance(tags, list):
                    results.append(", ".join(str(t) for t in tags))
                else:
                    results.append(str(tags))
            return results

        elif isinstance(self._strategy, (FastVLM, MoondreamVLM)):
            if len(extracted_images) > 1:
                batch_result = self._strategy.run_batch(extracted_images, extracted_prompt)
                # Se o batch já retornar lista, converte os itens. Se não, envelopa.
                if isinstance(batch_result, list):
                    return [str(r) for r in batch_result]
                return [str(batch_result)]
            
            # Execução de imagem única
            single_payload = {"image": extracted_images[0], "prompt": extracted_prompt}
            single_result = self._strategy.run(single_payload)
            return [str(single_result)]

    async def describe_person_faces(self, image: Image.Image) -> str:
        """Decoupled logic to describe faces using VLM without saving temp files."""
        descriptions = []
        
        strict_prompt = (
            "Act as a strict facial feature extractor. Describe the person in the image in a single, short sentence. "
            "Focus ONLY on: gender, hair color/style, eye color, and visible accessories. "
            "CRITICAL: Do NOT guess emotions. Do NOT describe the background. "
            "Example: Man with brown hair and green eyes, wearing round gold glasses."
        )

        for emb, face_crop in self.face_service.get_faces_embeddings(image):
            try:
                pil_crop = Image.fromarray(face_crop)
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                description = ""

                # 1. Grupo Síncrono
                if isinstance(self._strategy, (FastVLM, MoondreamVLM)):
                    payload = {"image": img_b64, "prompt": strict_prompt}
                    description = self._strategy.run(payload)

                # 2. Grupo Assíncrono (Padrão OpenAI)
                elif isinstance(self._strategy, (GGUF, LLMApi)):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": strict_prompt}
                        ]
                    }]
                    
                    if isinstance(self._strategy, LLMApi):
                        description = await self._strategy.run_chat(self.model_tag, messages, stream=False)
                    else:
                        description = await self._strategy.run_chat(messages, stream=False)
                
                else:
                    logger.warning(f"Strategy {type(self._strategy)} não suportada para extração de face.")
                    continue

                # 3. Limpeza
                if description:
                    description = self._extract_text(description)
                    clean_desc = description.replace('\n', ' ').strip()
                    descriptions.append(clean_desc)

            except Exception as e:
                logger.error(f"Erro ao gerar descrição do rosto com VLM: {e}")

        return " | ".join(descriptions) if descriptions else ""

    def unload(self):
        """Libera os recursos da estratégia (VLM, GGUF, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload(self.model_tag)
            logger.info(f"Eyes: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Eyes: Strategy {self.model_tag} cleared.")