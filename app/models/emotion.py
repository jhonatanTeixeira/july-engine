from __future__ import annotations
import os
import logging
from PIL import Image
from typing import Any, Dict, Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    import cv2
    import numpy as np
    import onnxruntime as ort

logger = logging.getLogger("JulyEngine.Models.Emotion")

class EmotionModel:
    def __init__(self, face_detector, backend="cpu"):
        self.backend = backend
        self.model_path = os.path.abspath(os.path.join("storage", "models", "emotion-ferplus-8.onnx"))
        self.session = None
        self.face_detector = face_detector

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Emotion roda na CPU (ONNX)."""
        return 0

    def load(self):
        if self.session is None:
            import onnxruntime as ort
            try:
                logger.info(f"Emotion: Loading model from {self.model_path}")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                logger.info("Emotion model loaded successfully.")
            except Exception as e:
                logger.error(f"Emotion: Failed to load: {e}")
                raise e

    def run(self, image: Union[Image.Image, List[Image.Image]]) -> Union[str, List[str]]:
        if self.session is None:
            self.load()

        if isinstance(image, list):
            return [self._run_single(img) for img in image]
        
        return self._run_single(image)

    def _run_single(self, image: Image.Image) -> str:
        import numpy as np
        import cv2
        
        img_rgb = np.array(image.convert('RGB'))
        faces = self.face_detector.detect_faces(img_rgb) if self.face_detector else []

        if not faces:
            # Se não detectou nada (ou não tem detector), assume que a imagem já é o recorte da face
            face_crop = img_rgb
        else:
            # Usamos a primeira face detectada para análise de emoção
            x1, y1, x2, y2 = faces[0]
            face_crop = img_rgb[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return "Empty face crop"
            
        try:
            # O modelo emotion-ferplus-8.onnx espera entrada (1, 1, 64, 64) em escala de cinza
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (64, 64))
            
            # Normalização e ajuste de shape
            input_tensor = face_resized.astype(np.float32).reshape(1, 1, 64, 64)
            
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            scores = outputs[0][0]
            dominant_emotion = emotions[np.argmax(scores)]
            
            logger.info(f"Emotion detected: {dominant_emotion}")
            return dominant_emotion
        except Exception as e:
            logger.error(f"Emotion execution failed: {e}")
            raise e
