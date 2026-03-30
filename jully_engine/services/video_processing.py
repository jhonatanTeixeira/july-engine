import abc
import asyncio
import base64
import datetime
import io
import logging
import os
import subprocess
from typing import Any, Generator, List, Optional
import uuid
from PIL import Image
import cv2
import numpy as np

from ..domain.entities import VideoAggregate, VideoSegment, GridNarrative
from .helpers import inference_helper
from ..services.vision import FaceService


logger = logging.getLogger("JulyEngine.Services.VideoProcessig")


class GridPackerService:
    """Responsável por fundir múltiplos frames em uma única imagem (Tiling/Grid)"""
    
    @staticmethod
    def pack_frames(frames: List[np.ndarray], grid_size: int = 2) -> Image.Image:
        """
        Recebe uma lista de frames (numpy BGR) e retorna uma imagem GRID (PIL RGB).
        Ex: grid_size=2 gera um grid 2x2 (4 frames).
        """
        if not frames: 
            return None
        
        # Redimensiona frames para garantir uniformidade no grid
        h, w, _ = frames[0].shape
        resized_frames = [cv2.resize(f, (w // 2, h // 2)) for f in frames[:grid_size*grid_size]]
        
        # Preenche com preto se houver menos frames que o grid
        while len(resized_frames) < grid_size * grid_size:
            resized_frames.append(np.zeros_like(resized_frames[0]))
            
        rows = []
        for i in range(0, len(resized_frames), grid_size):
            rows.append(np.hstack(resized_frames[i:i+grid_size]))
            
        grid_bgr = np.vstack(rows)
        grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(grid_rgb)


class VideoProcessorService:
    """Interface de Infraestrutura para extração de frames e bifurcação de stream"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def extract_segments(self, interval_sec: int = 2, frames_per_grid: int = 4):
        """
        Extrai frames em um buffer e retorna Grids para o VLM processar.
        interval_sec: Intervalo entre frames capturados.
        frames_per_grid: Quantos frames compõem um Grid (N).
        """
        hop = int(self.fps * interval_sec)
        frame_buffer = []
        timestamps = []
        count = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            if count % hop == 0:
                frame_buffer.append(frame)
                timestamps.append(count / self.fps)
                
                # Se o buffer atingiu N, despacha o Grid (Rota B)
                if len(frame_buffer) >= frames_per_grid:
                    grid_img = GridPackerService.pack_frames(frame_buffer)
                    yield timestamps[0], timestamps[-1], grid_img
                    frame_buffer = []
                    timestamps = []
            
            count += 1
        
        # Despacha o resto se sobrar
        if frame_buffer:
            grid_img = GridPackerService.pack_frames(frame_buffer)
            yield timestamps[0], timestamps[-1], grid_img
            
        self.cap.release()


class OpenCVBifurcator:
    def __init__(self, temp_dir: str = "storage/temp_video"):
        # Agora ele só guarda configuração de infraestrutura (diretórios)
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def sample_frames(self, video_path: str, interval_sec: float) -> Generator[tuple, None, None]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: return
        
        hop = int(fps * interval_sec)
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % hop == 0:
                timestamp = count / fps
                yield timestamp, frame
            count += 1
        cap.release()

    def extract_audio_stream(self, video_path: str) -> Optional[str]:
        audio_path = os.path.join(self.temp_dir, f"{os.path.basename(video_path)}.wav")
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return audio_path
        except Exception as e:
            logger.error(f"FFmpeg audio extraction failed: {e}")
            return None


class NumPyGridPacker:
    def pack(self, frames: List[np.ndarray], layout: tuple = (2, 2)) -> Image.Image:
        if not frames: return None
        
        grid_rows, grid_cols = layout
        n_required = grid_rows * grid_cols
        
        # Redimensiona frames para garantir uniformidade no grid (ex: 512x512 por slot)
        # Usamos o tamanho do primeiro frame como base, mas escalonado para baixo
        h, w, _ = frames[0].shape
        slot_h, slot_w = h // 2, w // 2
        
        resized_frames = [cv2.resize(f, (slot_w, slot_h)) for f in frames[:n_required]]
        
        # Preenche slots vazios com preto se necessário
        while len(resized_frames) < n_required:
            resized_frames.append(np.zeros((slot_h, slot_w, 3), dtype=np.uint8))
            
        rows = []
        for i in range(0, n_required, grid_cols):
            row = np.hstack(resized_frames[i:i+grid_cols])
            rows.append(row)
            
        grid_bgr = np.vstack(rows)
        grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(grid_rgb)


class VLMAdapter:
    
    def _img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    async def describe_grids_batch(self, grids: List[Image.Image], prompt: str) -> List[GridNarrative]:
        if not grids:
            return []

        # Montando o payload multimodal com N imagens no mesmo request
        content_array = [{"type": "text", "text": prompt}]
        
        for grid in grids:
            content_array.append({
                "type": "image_url", 
                "image_url": {"url": self._img_to_base64(grid)}
            })

        payload = {
            "messages": [
                {"role": "user", "content": content_array}
            ],
            "headers": {}
        }
        
        # Um único request para a GPU mastigar todas as imagens!
        results = await inference_helper.process('vision_chat', payload)
        
        # Assumindo que o fastvlm devolve uma lista de textos, um para cada imagem do batch
        narratives = []
        
        # Se o modelo não devolver lista, force para lista para parear com os grids
        if isinstance(results, str):
            results = [results]
            
        for description in results:
            desc_text = description.get("text", "") if isinstance(description, dict) else str(description)
            narratives.append(GridNarrative(
                text=desc_text,
                visual_vibe="dynamic", # Placeholder para expansão futura
                action_summary=desc_text[:100], 
                tokens_consumed=0 
            ))
            
        return narratives


class IVideoAnalysisStrategy(abc.ABC):
    @abc.abstractmethod
    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: Any) -> List[VideoSegment]:
        pass


class ObjectInteractionStrategy(IVideoAnalysisStrategy):
    def __init__(self, yolo_model_path='yolov8n.pt'):
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)

    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: Any) -> List[VideoSegment]:
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

class MultimodalVideoAnalysisUseCase:
    def __init__(
        self,
        bifurcator: OpenCVBifurcator,
        packer: NumPyGridPacker,
        vlm: VLMAdapter,
    ):
        self.bifurcator = bifurcator
        self.packer = packer
        self.vlm = vlm
        self.face_service = FaceService()

    async def execute(self, video_path: str, interval_sec: float = 2.0, frames_per_grid: int = 4, batch_size: int = 10, strategy: str = "default"):
        logger.info(f"Starting multimodal native batch analysis for: {video_path} with strategy: {strategy}")
        
        video_aggregate = VideoAggregate(
            video_id=str(uuid.uuid4()),
            file_path=video_path,
            metadata={"interval_sec": interval_sec, "strategy": strategy},
            processed_at=datetime.now()
        )

        # =====================================================================
        # DISPARO ASSÍNCRONO DA TRANSCRIÇÃO (Passando o video_path)
        # =====================================================================
        audio_task = asyncio.create_task(self._process_full_audio(video_path))

        # =====================================================================
        # PROCESSAMENTO VISUAL 
        # =====================================================================
        frame_buffer = []
        timestamps = []
        grid_batch = []
        timestamp_batch = [] 
        
        analysis_strategy = None
        if strategy == "interaction":
            analysis_strategy = ObjectInteractionStrategy()
        elif strategy == "emotion":
            analysis_strategy = EmotionAndAttentionStrategy()

        # Passando o video_path para o gerador
        for ts, frame in self.bifurcator.sample_frames(video_path, interval_sec):
            frame_buffer.append(frame)
            timestamps.append(ts)
            
            if len(frame_buffer) >= frames_per_grid:
                grid_img = self.packer.pack(frame_buffer)
                
                grid_batch.append(grid_img)
                timestamp_batch.append((timestamps[0], timestamps[-1]))
                
                frame_buffer = []
                timestamps = []
                
                if len(grid_batch) >= batch_size:
                    await self._process_batch(video_aggregate, grid_batch, timestamp_batch, analysis_strategy)
                    grid_batch = []
                    timestamp_batch = []

        if frame_buffer:
            grid_img = self.packer.pack(frame_buffer)
            grid_batch.append(grid_img)
            timestamp_batch.append((timestamps[0], timestamps[-1]))
            
        if grid_batch:
            await self._process_batch(video_aggregate, grid_batch, timestamp_batch, analysis_strategy)

        # =====================================================================
        # SINCRONIZAÇÃO FINAL
        # =====================================================================
        logger.info(f"Vision processing complete. Waiting for audio transcription to finish...")
        full_text = await audio_task
        video_aggregate.full_transcription = full_text

        logger.info(f"Multimodal Analysis complete. {len(video_aggregate.segments)} segments batched and created.")

        return video_aggregate

    async def _process_batch(self, aggregate: VideoAggregate, grids: List, time_ranges: List[tuple], strategy: Optional[Any] = None):
        if strategy:
            segments = await strategy.analyze_batch(grids, time_ranges, self.face_service, self.vlm)
            for seg in segments:
                # Map VideoSegment from vision.py to VideoSegment from entities.py
                new_seg = VideoSegment(
                    segment_id=str(uuid.uuid4()),
                    start_offset=seg.time_range[0],
                    end_offset=seg.time_range[1],
                    narrative=GridNarrative(text=seg.description)
                )
                aggregate.segments.append(new_seg)
        else:
            prompt = "Describe the sequence of actions, environment and people in this video narrative block."
            narratives = await self.vlm.describe_grids_batch(grids, prompt)
            
            for i, narrative in enumerate(narratives):
                start_ts, end_ts = time_ranges[i]
                
                segment = VideoSegment(
                    segment_id=str(uuid.uuid4()),
                    start_offset=start_ts,
                    end_offset=end_ts,
                    narrative=narrative
                )
                
                aggregate.segments.append(segment)

    async def _process_full_audio(self, video_path: str) -> str:
        """Extrai o .wav do vídeo e manda para o STT local."""
        try:
            audio_path = self.bifurcator.extract_audio_stream(video_path)
            
            if not audio_path or not os.path.exists(audio_path):
                logger.warning("No audio stream found or extraction failed.")
                return ""

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            payload = {
                "audio": audio_bytes,
                "headers": {} 
            }
            
            logger.info("Sending extracted audio to STT engine...")
            
            transcription = await inference_helper.process('stt', payload)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Audio transcription failed during video analysis: {e}")
            return ""
        finally:
            if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)


multimodal_video_analysis = MultimodalVideoAnalysisUseCase(OpenCVBifurcator(), NumPyGridPacker(), VLMAdapter())