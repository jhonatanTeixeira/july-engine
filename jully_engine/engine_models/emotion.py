import os
import logging
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Any, Dict, Optional

logger = logging.getLogger("JulyEngine.Models.Emotion")

class Emotion:
    def __init__(self, backend="cpu"):
        self.backend = backend
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "emotion-ferplus-8.onnx"))
        self.session = None

    def load(self):
        if self.session is None:
            try:
                logger.info(f"Emotion: Loading model from {self.model_path}")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                logger.info("Emotion model loaded successfully.")
            except Exception as e:
                logger.error(f"Emotion: Failed to load: {e}")
                raise e

    def _preprocess(self, image: Image.Image):
        # Convert PIL to cv2 format
        img_np = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(30, 30))
        
        if len(faces) == 0:
            # Try alt2 if default fails
            alt_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            if os.path.exists(alt_cascade_path):
                alt_cascade = cv2.CascadeClassifier(alt_cascade_path)
                faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        # Prepare tensor
        img_data = np.array(face_resized).astype(np.float32)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    def run(self, image: Image.Image) -> str:
        if self.session is None:
            self.load()
            
        input_data = self._preprocess(image)
        if input_data is None:
            return "No face detected"
            
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})
            
            emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
            scores = outputs[0][0]
            dominant_emotion = emotions[np.argmax(scores)]
            return dominant_emotion
        except Exception as e:
            logger.error(f"Emotion execution failed: {e}")
            raise e
