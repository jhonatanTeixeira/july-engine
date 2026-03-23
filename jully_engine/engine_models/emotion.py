import os
import logging
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Any, Dict, Optional

logger = logging.getLogger("JulyEngine.Models.Emotion")

class Emotion:
    def __init__(self, face_detector, backend="cpu"):
        self.backend = backend
        self.model_path = os.path.abspath(os.path.join("storage", "models", "emotion-ferplus-8.onnx"))
        self.session = None
        self.face_detector = face_detector

    def load(self):
        if self.session is None:
            try:
                logger.info(f"Emotion: Loading model from {self.model_path}")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                logger.info("Emotion model loaded successfully.")
            except Exception as e:
                logger.error(f"Emotion: Failed to load: {e}")
                raise e

    def run(self, image: Image.Image) -> str:
        if self.session is None:
            self.load()

        img_rgb = np.array(image.convert('RGB'))
        input_data = self.face_detector.detect_faces(img_rgb)

        if input_data is None:
            return "No face detected"
            
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})
            
            emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            scores = outputs[0][0]
            dominant_emotion = emotions[np.argmax(scores)]
            logger.info(f"Engine Emotion executed successfully on {self.backend} with Emotion")
            return dominant_emotion
        except Exception as e:
            logger.error(f"Emotion execution failed: {e}")
            raise e
