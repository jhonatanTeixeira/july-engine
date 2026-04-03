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

if TYPE_CHECKING:
    import cv2
    import numpy as np
    import mediapipe as mp
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
    def __init__(self, model_path='storage/models/detector.tflite', min_confidence=0.3):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_confidence)
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, img_rgb: "np.ndarray") -> List[Tuple[int, int, int, int]]:
        import mediapipe as mp
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
        
    def get_faces_embeddings(self, image: Image.Image) -> Generator[Tuple[List[float], "np.ndarray", Tuple[int, int, int, int]], None, None]:
        import numpy as np
        import cv2 # Necessário para a conversão de cores
        
        # Mantém RGB para o MediaPipe
        img_np = np.array(image.convert('RGB'))
        faces_coords = self.detector.detect_faces(img_np)

        for (x1, y1, x2, y2) in faces_coords:
            margin_w = int((x2 - x1) * 0.2)
            margin_h = int((y2 - y1) * 0.2)
            
            x1_m = max(0, x1 - margin_w)
            y1_m = max(0, y1 - margin_h)
            x2_m = min(img_np.shape[1], x2 + margin_w)
            y2_m = min(img_np.shape[0], y2 + margin_h)
            
            face_crop_rgb = img_np[y1_m:y2_m, x1_m:x2_m]

            if face_crop_rgb.size == 0: 
                continue

            # 2. A CORREÇÃO MÁGICA: Converte de RGB para BGR para o DeepFace ver as cores certas
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)

            try:
                from deepface import DeepFace
                rep = DeepFace.represent(
                    img_path=face_crop_bgr, # Passando a matriz BGR!
                    model_name="ArcFace",
                    detector_backend="opencv", 
                    enforce_detection=True, 
                    align=False # 3. Desligamos o alinhamento para parar de deitar as fotos
                )
                # Devolve o face_crop_rgb para o frontend exibir a foto com as cores normais
                yield rep[0]["embedding"], face_crop_rgb, (x1, y1, x2, y2)
                
            except ValueError:
                # O OpenCV barrou o falso positivo (ex: flor, gato)
                pass
            except Exception as e:
                logger.error(f"Erro inesperado no DeepFace: {e}")

    def extract_and_match_faces(self, image: Image.Image, pic_id: str, collection: str = "faces_embeddings") -> List[str]:
        """Detect faces in a single image and match them."""
        res = self.sync_faces_batch([image], [pic_id], collection=collection)
        return res[0] if res else []

    def match_or_add_face(self, emb: List[float], pic_id: str, collection: str = "faces_embeddings") -> Tuple[str, List[float]]:
        try:
            matches = self.vector_store.search_with_details(query_embedding=emb, top_k=1, collection=collection)
            if matches:
                match = matches[0]
                import numpy as np
                
                if match['distance'] < 0.72:
                    doc_id = match['id'] 
                    person_id = match.get('metadata', {}).get('person_id') or str(uuid.uuid4())
                    
                    vetor_antigo = np.array(match['embedding'])
                    vetor_novo = np.array(emb)
                    
                    vetor_mesclado = (vetor_antigo * 0.85) + (vetor_novo * 0.15)
                    vetor_normalizado = vetor_mesclado / np.linalg.norm(vetor_mesclado)
                    
                    novo_emb_list = vetor_normalizado.tolist()
                    
                    self.vector_store.update_embedding(doc_id=doc_id, new_embedding=novo_emb_list, collection=collection)
                    return person_id, novo_emb_list
            
            person_id = str(uuid.uuid4())
            self.vector_store.add(
                text="", 
                embedding=emb, 
                metadata={"person_id": person_id, "pic_id": str(pic_id), "collection": collection},
                collection=collection 
            )
            return person_id, emb
        except Exception as e:
            logger.error(f"Erro no match_or_add_face: {e}")
            return str(uuid.uuid4()), emb

    def sync_faces_batch(self, images: List[Image.Image], pic_ids: List[str], collection: str = "faces_embeddings") -> List[List[Dict[str, Any]]]:
        """
        Orquestrador em lote para processar imagens e persistir no RAG.
        Retorna: List[List[dict]] - Cada item contém: person_id, embedding e face_b64.
        """
        results = []
        for img, pic_id in zip(images, pic_ids):
            faces_found = []
            if img:
                for emb, face_crop, _ in self.get_faces_embeddings(img):
                    # O match_or_add_face já cuida da persistência/atribuição de ID na Engine
                    p_id, final_emb = self.match_or_add_face(emb, pic_id, collection=collection)
                    
                    # Converte o recorte para Base64 para o Photos salvar a imagem
                    try:
                        pil_crop = Image.fromarray(face_crop)
                        buffered = io.BytesIO()
                        pil_crop.save(buffered, format="JPEG")
                        face_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    except Exception as e:
                        logger.error(f"Erro ao encodar crop de rosto em sync_batch: {e}")
                        face_b64 = ""

                    faces_found.append({
                        "person_id": str(p_id),
                        "embedding": list(final_emb) if hasattr(final_emb, "tolist") else final_emb,
                        "face_b64": face_b64
                    })
            results.append(faces_found)
        return results


class BatchProcessingService:
    def __init__(self, face_service, rag_strategy):
        from ..persistence.persistence import get_backend
        self.face_service = face_service
        self.rag_strategy = rag_strategy
        self.db = get_backend()

    async def process_batch(self, filenames: List[str]):
        from .helpers import inference_helper
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

        # Send ENTIRE batch of valid images to the VLM
        try:
            messages = [{"role": "user", "content": []}]
            for b64 in b64_images:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            messages[0]["content"].append({
                "type": "text", 
                "text": f"{prompt}\nPlease provide a description for each image in order, separated by a newline."
            })

            raw_descriptions = await inference_helper.process('vision_chat', {
                "messages": messages, 
                "model": "fastvlm"
            })
            
            def _extract_text(response: Any) -> str:
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
                return str(response)

            descriptions = []
            if isinstance(raw_descriptions, list):
                descriptions = [_extract_text(d) for d in raw_descriptions]
            else:
                text_res = _extract_text(raw_descriptions)
                descriptions = [d.strip() for d in text_res.split('\n') if d.strip()]
                
            while len(descriptions) < len(valid_images):
                descriptions.append("No description.")
                
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
        batch_face_embs = self.face_service.sync_faces_batch(valid_images, pic_ids)

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


class VideoSegment:
    def __init__(self, time_range: tuple, description: str):
        self.time_range = time_range
        self.description = description

class IVideoAnalysisStrategy(abc.ABC):
    @abc.abstractmethod
    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: Any) -> List[VideoSegment]:
        pass

class ObjectInteractionStrategy(IVideoAnalysisStrategy):
    def __init__(self, yolo_model_path='yolov8n.pt'):
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)

    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: Any) -> List[VideoSegment]:
        from .helpers import inference_helper
        segments = []
        
        batch_face_bboxes = []
        for grid in grids:
            faces = []
            for emb, face_crop, bbox in face_service.get_faces_embeddings(grid):
                faces.append(bbox)
            batch_face_bboxes.append(faces)

        results = self.yolo.predict(grids, verbose=False)
        batch_object_bboxes = []
        for r in results:
            objs = []
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                objs.append({"bbox": box.tolist(), "class_name": self.yolo.names[int(cls)]})
            batch_object_bboxes.append(objs)

        b64_grids = []
        for grid in grids:
            buffered = io.BytesIO()
            grid.save(buffered, format="JPEG")
            b64_grids.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        messages = [{"role": "user", "content": []}]
        for b64 in b64_grids:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        spatial_metadata = ""
        for i, (f_boxes, o_boxes) in enumerate(zip(batch_face_bboxes, batch_object_bboxes)):
            spatial_metadata += f"\nImage {i+1}:\n"
            for fb in f_boxes:
                spatial_metadata += f"  Person face at {fb}\n"
            for ob in o_boxes:
                spatial_metadata += f"  Object '{ob['class_name']}' at {ob['bbox']}\n"

        prompt = (
            f"Here are {len(grids)} images and their spatial bounding boxes for faces and objects.\n"
            f"{spatial_metadata}\n"
            "Analyze the spatial proximity between the people's faces and objects (e.g. objects near their hands/bottom of face). "
            "For each image, provide a description in the format: 'Person [person_id] with [face_description] is holding [Object]'. "
            "Separate each image description by a newline."
        )
        
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        try:
            res = await inference_helper.process('vision_chat', {"messages": messages, "model": "fastvlm"})
            
            def _extract_text(response: Any) -> str:
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
                return str(response)

            text_response = _extract_text(res)
            descriptions = [d.strip() for d in text_response.split('\n') if d.strip()]
            
            while len(descriptions) < len(grids):
                descriptions.append("No clear interaction.")

            for tr, desc in zip(time_ranges, descriptions):
                segments.append(VideoSegment(tr, desc))

        except Exception as e:
            logger.error(f"Error in ObjectInteractionStrategy: {e}")
            for tr in time_ranges:
                segments.append(VideoSegment(tr, "Error processing interaction."))

        return segments

class EmotionAndAttentionStrategy(IVideoAnalysisStrategy):
    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: Any) -> List[VideoSegment]:
        from .helpers import inference_helper
        segments = []
        from ..engine_models.emotion import Emotion
        emotion_model = Emotion(face_service.detector, backend="cpu")

        batch_emotions = []
        for grid in grids:
            emotions = []
            for emb, face_crop, bbox in face_service.get_faces_embeddings(grid):
                pil_crop = Image.fromarray(face_crop)
                emotion_res = emotion_model.run({"image": pil_crop, "prompt": "detect emotion"})
                emotions.append(str(emotion_res))
            batch_emotions.append(emotions)

        b64_grids = []
        for grid in grids:
            buffered = io.BytesIO()
            grid.save(buffered, format="JPEG")
            b64_grids.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        messages = [{"role": "user", "content": []}]
        for b64 in b64_grids:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        
        prompt = (
            "Analyze these images. For each image, deduce the body posture and gaze direction of the people visible. "
            "Return exactly one sentence per image describing the posture and attention/gaze, separated by a newline."
        )
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        try:
            res = await inference_helper.process('vision_chat', {"messages": messages, "model": "fastvlm"})
            
            def _extract_text(response: Any) -> str:
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
                return str(response)

            text_response = _extract_text(res)
            postures = [d.strip() for d in text_response.split('\n') if d.strip()]
            
            while len(postures) < len(grids):
                postures.append("Posture unknown.")

            for i, tr in enumerate(time_ranges):
                emos = ", ".join(batch_emotions[i]) if batch_emotions[i] else "neutral"
                desc = f"Emotions: {emos}. Posture & Gaze: {postures[i]}"
                segments.append(VideoSegment(tr, desc))

        except Exception as e:
            logger.error(f"Error in EmotionAndAttentionStrategy: {e}")
            for tr in time_ranges:
                segments.append(VideoSegment(tr, "Error processing emotion/attention."))

        return segments
