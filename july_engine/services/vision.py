from __future__ import annotations
import abc
import asyncio
import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Generator, Optional, Union, TYPE_CHECKING
from PIL import Image

import numpy as np
import cv2

if TYPE_CHECKING:
    from deepface import DeepFace

from ..persistence.vector_store import vector_store

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
        from exif import Image as ExifImage
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
    def __init__(self, model_path='yolo11s.pt', min_confidence=0.3):
        from ultralytics import YOLO
        # O usuário solicitou reforçar o uso de yolov11s
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"Falha ao carregar YOLOv11s: {e}. Tentando yolo11n.pt como fallback.")
            self.model = YOLO('yolo11n.pt')
        self.min_confidence = min_confidence

    def detect_faces(self, img_rgb: "np.ndarray") -> List[Tuple[int, int, int, int]]:
        """
        Detecta bboxes no frame. 
        Nota: Para faces especificamente, se o yolov11s padrão não for suficiente, 
        o FaceService já usa o DeepFace com detector_backend='yolov11s'.
        """
        results = self.model(img_rgb, conf=self.min_confidence, verbose=False)
        faces = []
        for r in results:
            for box in r.boxes:
                # No YOLO padrão, faces não são uma classe. 
                # Se o usuário forneceu um modelo de face, pegamos todas as detecções.
                # Se for o modelo COCO, pegamos 'person' (classe 0).
                if int(box.cls[0]) == 0 or len(self.model.names) < 10: # heurística para face-model vs coco-model
                    b = box.xyxy[0].cpu().numpy()
                    faces.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return faces

class FaceService:
    def __init__(self):
        from ..persistence.vector_store import vector_store

        self.vector_store = vector_store
        self.model_name = "ArcFace"
        self.detector_backend = "yolov11s"
        self.detector = FaceDetector()

    def get_faces_embeddings(self, image: Image.Image) -> Generator[Tuple[List[float], "np.ndarray", Tuple[int, int, int, int]], None, None]:
        """
        Detecta e vetoriza faces diretamente via DeepFace usando YOLOv11s.
        """
        # 1. Preparação: DeepFace trabalha melhor com BGR (OpenCV Style)
        img_rgb = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        try:
            # 2. Execução Unificada: O YOLOv11s varre a imagem inteira aqui
            results = DeepFace.represent(
                img_path=img_bgr,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=False  # Mantido False conforme sua regra para não deitar fotos
            )

            for rep in results:
                embedding = rep["embedding"]
                area = rep["facial_area"] # x, y, w, h
                
                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                
                # 3. Aplicar Margem (20%) para o recorte que vai para o Frontend
                margin_w = int(w * 0.2)
                margin_h = int(h * 0.2)
                
                y1 = max(0, y - margin_h)
                y2 = min(img_rgb.shape[0], y + h + margin_h)
                x1 = max(0, x - margin_w)
                x2 = min(img_rgb.shape[1], x + w + margin_w)
                
                # Recorte em RGB para exibição correta no UI
                face_crop_rgb = img_rgb[y1:y2, x1:x2]

                if face_crop_rgb.size > 0:
                    yield embedding, face_crop_rgb, (x, y, x + w, y + h)

        except ValueError:
            # DeepFace lança ValueError se enforce_detection=True e não achar nada
            pass
        except Exception as e:
            logger.error(f"Erro inesperado no DeepFace (YOLOv11s): {e}")

    def match_or_add_face(self, emb: List[float], pic_id: str, collection: str = "faces_embeddings") -> Tuple[str, List[float]]:
        """Busca no RAG, aplica EMA no vetor ou cria nova identidade."""
        try:
            matches = self.vector_store.search_with_details(
                query_embedding=emb, 
                top_k=1, 
                collection=collection, 
                model_tag="arcface"
            )
            
            if matches:
                match = matches[0]
                # Threshold de 0.72 conforme sua implementação
                if match['distance'] < 0.72:
                    doc_id = match['id'] 
                    person_id = match.get('metadata', {}).get('person_id') or str(uuid.uuid4())
                    
                    vetor_antigo = np.array(match['embedding'])
                    vetor_novo = np.array(emb)
                    
                    # 4. EMA (Exponential Moving Average): 85% antigo / 15% novo
                    # Isso evita o "apodrecimento" do vetor por variações bruscas
                    vetor_mesclado = (vetor_antigo * 0.85) + (vetor_novo * 0.15)
                    vetor_normalizado = vetor_mesclado / np.linalg.norm(vetor_mesclado)
                    
                    novo_emb_list = vetor_normalizado.tolist()
                    
                    self.vector_store.update_embedding(
                        doc_id=doc_id, 
                        new_embedding=novo_emb_list, 
                        collection=collection, 
                        model_tag="arcface"
                    )
                    return person_id, novo_emb_list
            
            # Nova Identidade
            person_id = str(uuid.uuid4())
            self.vector_store.add(
                text="", 
                embedding=emb, 
                metadata={"person_id": person_id, "pic_id": str(pic_id), "collection": collection},
                collection=collection,
                model_tag="arcface"
            )
            return person_id, emb
        except Exception as e:
            logger.error(f"Erro no match_or_add_face: {e}")
            return str(uuid.uuid4()), emb

    def sync_faces_batch(self, images: List[Image.Image], pic_ids: List[str], collection: str = "faces_embeddings") -> List[List[Dict[str, Any]]]:
        """Processa lote de imagens e retorna metadados para persistência no banco relacional."""
        results = []
        for img, pic_id in zip(images, pic_ids):
            faces_found = []
            if img:
                # get_faces_embeddings agora usa o YOLOv11s interno do DeepFace
                for emb, face_crop, _ in self.get_faces_embeddings(img):
                    p_id, final_emb = self.match_or_add_face(emb, pic_id, collection=collection)
                    
                    try:
                        pil_crop = Image.fromarray(face_crop)
                        buffered = io.BytesIO()
                        pil_crop.save(buffered, format="JPEG")
                        face_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    except Exception as e:
                        logger.error(f"Erro ao encodar crop: {e}")
                        face_b64 = ""

                    faces_found.append({
                        "person_id": str(p_id),
                        "embedding": list(final_emb) if hasattr(final_emb, "tolist") else final_emb,
                        "face_b64": face_b64
                    })
            results.append(faces_found)
        return results
