from datetime import datetime
import logging
import os
import uuid

import cv2
import numpy as np
import mediapipe as mp

from exif import Image as ExifImage
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
from PIL import Image

from ..engine_models.fastvlm import FastVLM
from ..domain.vector_store import vector_store

logger = logging.getLogger("JulyEngine.Services.Vision")


class AppConfig:
    DIR_PATH = r'E:\bkp\fotos bkp\Camera'
    FACES_PATH = r'E:\bkp\fotos bkp\public\faces'
    MONGO_URI = 'mongodb://localhost:27017/'
    DB_NAME = 'photos_bkp'
    BATCH_SIZE = 10


class ExifService:
    @staticmethod
    def extract_metadata(file_path):
        """ Extrai metadatos EXIF (Data e GPS) usando a biblioteca exif """
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
                        degrees = dms_tuple[0]
                        minutes = dms_tuple[1]
                        seconds = dms_tuple[2]
                        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                        if ref in ['S', 'W']:
                            decimal = -decimal
                        return decimal

                    lat = dms_to_decimal(lat_tuple, lat_ref)
                    lon = dms_to_decimal(lon_tuple, lon_ref)
        except Exception as e:
            logger.error(f"Erro ao ler EXIF de {file_path}: {e}")
        return date_val, lat, lon


class FaceDetector:
    def __init__(self, model_path='storage/models/detector.tflite', min_confidence=0.2):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options, 
            min_detection_confidence=min_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, img_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.detector.detect(mp_image)

        faces = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height

            if (x2 - x1) > 15 and (y2 - y1) > 15:
                faces.append((x1, y1, x2, y2))
        return faces


class FaceService:
    def __init__(self, detector_model_path='storage/models/detector.tflite'):
        self.detector = FaceDetector(model_path=detector_model_path)
        self.vector_store = vector_store
        
    def get_faces_embeddings(self, image: Image.Image):
        faces_coords = self.detector.detect_faces(img_np := np.array(image.convert('RGB')))

        for (x1, y1, x2, y2) in faces_coords:
            face_crop = img_np[y1:y2, x1:x2]
            
            if face_crop.size == 0: 
                continue

            rep = DeepFace.represent(
                img_path=face_crop, 
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
                align=True
            )
        
            yield rep[0]["embedding"], face_crop

    def extract_and_match_faces(self, image, pic_id):
        """Orquestrador Stateless: Calcula o centróide e atualiza o VectorStore."""
        face_embs_list = []
        
        if image is not None:
            # O loop consome as duas variáveis do nosso gerador
            for emb, face_crop in self.get_faces_embeddings(image):
                try:
                    # 1. Busca puramente no VectorStore
                    matches = self.vector_store.search_with_details(query_embedding=emb, top_k=1)
                    is_new_face = True
                    
                    if matches:
                        match = matches[0]
                        distancia_encontrada = match['distance']
                        
                        # 2. Rosto Conhecido (Threshold < 0.60)
                        if distancia_encontrada < 0.60:
                            is_new_face = False
                            doc_id = match['id'] 
                            
                            # ATUALIZAÇÃO DO CENTRÓIDE (Pura Matemática)
                            vetor_antigo = np.array(match['embedding'])
                            vetor_atual = np.array(emb)
                            
                            # Suaviza a transição: 85% do rosto base + 15% do rosto novo
                            novo_vetor = (vetor_antigo * 0.85) + (vetor_atual * 0.15)
                            novo_emb_list = novo_vetor.tolist()
                            
                            # Atualiza APENAS no VectorStore
                            self.vector_store.update_embedding(doc_id=doc_id, new_embedding=novo_emb_list)
                            
                            face_embs_list.append(novo_emb_list)
                    
                    # 3. Rosto Desconhecido (Novo cadastro)
                    if is_new_face:
                        person_id = str(uuid.uuid4())
                        
                        # Usamos o face_crop que veio no yield para salvar o JPG!
                        cv2.imwrite(
                            os.path.join(AppConfig.FACES_PATH, f"{person_id}.jpg"), 
                            cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR) # Previne o rosto azul
                        )
                        
                        # Indexa no Banco Vetorial
                        self.vector_store.add(
                            text="", 
                            embedding=emb, 
                            metadata={"person_id": person_id, "pic_id": pic_id} 
                        )
                        
                        face_embs_list.append(emb)
                        
                except Exception as e:
                    logger.error(f"Erro ao processar matemática do rosto: {e}")
                    
        return face_embs_list

# TODO: review this service to be usefull on this system
class BatchProcessingService:
    def __init__(self, vision_strategy, face_service, rag_strategy, db, img_collection, face_collection):
        self.vision_strategy = vision_strategy
        self.face_service = face_service
        self.rag_strategy = rag_strategy
        self.db = db
        self.img_collection = img_collection
        self.face_collection = face_collection

    def process_batch(self, filenames):
        valid_paths, valid_filenames, skipped = [], [], []
        for filename in filenames:
            file_path = os.path.join(AppConfig.DIR_PATH, filename)
            if self.db.pictures.find_one({"path": file_path}):
                skipped.append(filename)
                continue
            try:
                Image.open(file_path).verify()
                valid_paths.append(file_path)
                valid_filenames.append(filename)
            except: 
                logger.error(f"Arquivo inválido ou corrompido: {filename}")

        if not valid_paths: return skipped, []

        prompt = "Describe this image in detail. Focus on people, their physical traits, clothing colors and the environment."
        try: 
            import asyncio
            # If running in async context, run it synchronously if it's not a coroutine, or await if it is.
            # But process_batch is sync right now. Since describe_batch is async, we need an event loop.
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if asyncio.iscoroutinefunction(self.vision_strategy.describe_batch):
                description_results = loop.run_until_complete(self.vision_strategy.describe_batch(valid_paths, prompt))
            else:
                description_results = self.vision_strategy.describe_batch(valid_paths, prompt)
        except Exception as e:
            logger.error(f"Erro fatal no Lote VLM (OOM?): {e}")
            return skipped, []

        processed = []
        for i, file_path in enumerate(valid_paths):
            filename = valid_filenames[i]
            description = description_results[i]
            
            date_val, lat, lon = ExifService.extract_metadata(file_path)
            desc_vector = self.rag_strategy.encode(description)
            
            pic_res = self.db.pictures.insert_one({
                "path": file_path, "file": filename, "description": description,
                "date": date_val, "lat": lat, "lon": lon, "processed": True
            })
            pic_id = pic_res.inserted_id

            face_embs_list = self.face_service.extract_and_match_faces(
                file_path, pic_id, self.db, self.face_collection
            )
            
            self.img_collection.add(
                ids=[str(pic_id)],
                embeddings=[desc_vector],
                metadatas=[{
                    "path": file_path, "description": description, "date": date_val or "",
                    "lat": float(lat), "lon": float(lon), "face_embeddings": json.dumps(face_embs_list)
                }]
            )
            processed.append(filename)

        return skipped, processed
