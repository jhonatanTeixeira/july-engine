import abc
import asyncio
import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Generator, Optional, Union

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from exif import Image as ExifImage
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

from ..persistence.vector_store import vector_store
from .helpers import inference_helper

logger = logging.getLogger("JulyEngine.Services.Vision")

class AppConfig:
    DIR_PATH = r'E:\bkp\fotos bkp\Camera'
    FACES_PATH = r'E:\bkp\fotos bkp\public\faces'
    MONGO_URI = 'mongodb://localhost:27017/'
    DB_NAME = 'photos_bkp'
    BATCH_SIZE = 10

class ExifService:
    @staticmethod
    def extract_metadata(file_path: str):
        date_val, lat, lon = None, 0.0, 0.0
        try:
            with open(file_path, 'rb') as image_file:
                my_image = ExifImage(image_file)
                if my_image.has_exif and hasattr(my_image, 'datetime_original'):
                    date_str = my_image.datetime_original
                    date_val = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S').isoformat()
                if my_image.has_exif and hasattr(my_image, 'gps_latitude') and hasattr(my_image, 'gps_longitude'):
                    lat_tuple = my_image.gps_latitude
                    lon_tuple = my_image.gps_longitude
                    lat_ref = my_image.gps_latitude_ref
                    lon_ref = my_image.gps_longitude_ref
                    
                    def dms_to_decimal(dms_tuple, ref):
                        degrees, minutes, seconds = dms_tuple
                        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                        if ref in ['S', 'W']: decimal = -decimal
                        return decimal

                    lat = dms_to_decimal(lat_tuple, lat_ref)
                    lon = dms_to_decimal(lon_tuple, lon_ref)
        except Exception as e:
            logger.error(f"Erro ao ler EXIF de {file_path}: {e}")
        return date_val, lat, lon


class FaceDetector:
    def __init__(self, model_path='storage/models/detector.tflite', min_confidence=0.2):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_confidence)
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.detector.detect(mp_image)
        faces = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x1, y1 = max(0, bbox.origin_x), max(0, bbox.origin_y)
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            if (x2 - x1) > 15 and (y2 - y1) > 15:
                faces.append((x1, y1, x2, y2))
        return faces


class FaceService:
    def __init__(self, detector_model_path='storage/models/detector.tflite'):
        self.detector = FaceDetector(model_path=detector_model_path)
        self.vector_store = vector_store
        
    def get_faces_embeddings(self, image: Image.Image) -> Generator[Tuple[List[float], np.ndarray, Tuple[int, int, int, int]], None, None]:
        img_np = np.array(image.convert('RGB'))
        faces_coords = self.detector.detect_faces(img_np)

        for (x1, y1, x2, y2) in faces_coords:
            face_crop = img_np[y1:y2, x1:x2]
            if face_crop.size == 0: 
                continue

            try:
                rep = DeepFace.represent(
                    img_path=face_crop, 
                    model_name="ArcFace",
                    detector_backend="skip",
                    enforce_detection=False,
                    align=True
                )
                yield rep[0]["embedding"], face_crop, (x1, y1, x2, y2)
            except Exception as e:
                logger.error(f"Error in deepface representation: {e}")

    def extract_and_match_faces(self, image, pic_id):
        # Keep backward compatibility
        res = self.extract_and_match_faces_batch([image], [pic_id])
        return res[0] if res else []

    def extract_and_match_faces_batch(self, images: List[Image.Image], pic_ids: List[str]) -> List[List[List[float]]]:
        """Batched orchestrator: extract faces from multiple images and match them."""
        results = []
        for img, pic_id in zip(images, pic_ids):
            face_embs_list = []
            if img is not None:
                for emb, face_crop, _ in self.get_faces_embeddings(img):
                    try:
                        matches = self.vector_store.search_with_details(query_embedding=emb, top_k=1)
                        is_new_face = True
                        
                        if matches:
                            match = matches[0]
                            if match['distance'] < 0.60:
                                is_new_face = False
                                doc_id = match['id'] 
                                novo_vetor = (np.array(match['embedding']) * 0.85) + (np.array(emb) * 0.15)
                                novo_emb_list = novo_vetor.tolist()
                                self.vector_store.update_embedding(doc_id=doc_id, new_embedding=novo_emb_list)
                                face_embs_list.append(novo_emb_list)
                        
                        if is_new_face:
                            person_id = str(uuid.uuid4())
                            os.makedirs(AppConfig.FACES_PATH, exist_ok=True)
                            cv2.imwrite(
                                os.path.join(AppConfig.FACES_PATH, f"{person_id}.jpg"), 
                                cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                            )
                            self.vector_store.add(
                                text="", 
                                embedding=emb, 
                                metadata={"person_id": person_id, "pic_id": str(pic_id)} 
                            )
                            face_embs_list.append(emb)
                    except Exception as e:
                        logger.error(f"Erro ao processar matemática do rosto: {e}")
            results.append(face_embs_list)
        return results

class BatchProcessingService:
    def __init__(self, face_service, rag_strategy):
        from ..persistence.persistence import get_backend
        self.face_service = face_service
        self.rag_strategy = rag_strategy
        self.db = get_backend()

    async def process_batch(self, filenames: List[str]):
        valid_images, valid_paths, valid_filenames, skipped = [], [], [], []

        for filename in filenames:
            file_path = os.path.join(AppConfig.DIR_PATH, filename)
            # Use persistence db abstract backend
            if self.db.find_one("pictures", {"path": file_path}):
                skipped.append(filename)
                continue
            try:
                img = Image.open(file_path).convert('RGB')
                valid_images.append(img)
                valid_paths.append(file_path)
                valid_filenames.append(filename)
            except Exception as e:
                logger.error(f"Arquivo inválido ou corrompido: {filename} - {e}")

        if not valid_images: return skipped, []

        prompt = "Describe this image in detail. Focus on people, their physical traits, clothing colors and the environment."
        
        b64_images = []
        for img in valid_images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            b64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        # Send ENTIRE batch of valid images to the VLM (inference_helper supports vision_chat with multiline/multiple messages or multiple image URLs if fastvlm handles it, here we assume it can map or inference_helper process batched)
        try:
            messages = [{"role": "user", "content": []}]
            for b64 in b64_images:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            # Adiciona o prompt no final do array de imagens
            messages[0]["content"].append({
                "type": "text", 
                "text": f"{prompt}\nPlease provide a description for each image in order, separated by a newline."
            })

            # UMA ÚNICA chamada pro Orquestrador com N imagens!
            raw_descriptions = await inference_helper.process('vision_chat', {
                "messages": messages, 
                "model": "fastvlm"
            })
            
            def _extract_text(response: Any) -> str:
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
                return str(response)

            descriptions = []
            for d in raw_descriptions:
                if isinstance(d, Exception):
                    logger.error(f"Erro no Lote VLM: {d}")
                    descriptions.append("")
                else:
                    descriptions.append(_extract_text(d))
        except Exception as e:
            logger.error(f"Erro fatal no Lote VLM: {e}")
            return skipped, []

        docs_to_insert = []
        for i, file_path in enumerate(valid_paths):
            date_val, lat, lon = ExifService.extract_metadata(file_path)
            doc = {
                "id": str(uuid.uuid4()),
                "path": file_path, 
                "file": valid_filenames[i], 
                "description": descriptions[i],
                "date": date_val, 
                "lat": lat, 
                "lon": lon, 
                "processed": True
            }
            docs_to_insert.append(doc)

        # Batch insert DB
        self.db.insert_many("pictures", docs_to_insert)

        # Extrato batched faces
        pic_ids = [doc["id"] for doc in docs_to_insert]
        batch_face_embs = self.face_service.extract_and_match_faces_batch(valid_images, pic_ids)

        processed = []
        for i, doc in enumerate(docs_to_insert):
            desc_vector = self.rag_strategy.encode(doc["description"])
            vector_store.add(
                text=doc["description"],
                embedding=desc_vector,
                metadata={
                    "pic_id": doc["id"],
                    "path": doc["path"], 
                    "date": doc["date"] or "",
                    "lat": float(doc["lat"]), 
                    "lon": float(doc["lon"]), 
                    "face_embeddings": json.dumps(batch_face_embs[i])
                }
            )
            processed.append(doc["file"])

        return skipped, processed
