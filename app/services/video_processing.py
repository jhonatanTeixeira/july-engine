import abc
import asyncio
import base64
from datetime import datetime
import io
import logging
import os
import subprocess
from typing import Any, Dict, Generator, List, Optional
import uuid
from PIL import Image
import cv2
import numpy as np

from ..domain.entities import VideoAggregate, VideoSegment, GridNarrative
from ..services.vision import FaceService


logger = logging.getLogger("JulyEngine.Services.VideoProcessing")


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


class OpenCVBifurcator:
    def __init__(self, temp_dir: str = "storage/temp_video"):
        # Agora ele só guarda configuração de infraestrutura (diretórios)
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def sample_frames(self, video_path: str, interval_sec: float, scene_threshold: float = 0.85, detect_change=False) -> Generator[tuple, None, None]:
        """
        Extrai frames em intervalos regulares e filtra frames com mudanças insignificantes.
        """
        logger.info(f'sampling frames with scene-change detection (threshold={scene_threshold})')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"OpenCV: Video opened. FPS: {fps}, Total Frames: {total_frames}, Path: {video_path}")
        
        if fps == 0:
            logger.error("OpenCV: FPS is 0, video could not be read correctly.")
            cap.release()
            return

        hop = int(fps * interval_sec)
        count = 0
        last_keyframe: Optional[np.ndarray] = None
        last_hist: Optional[np.ndarray] = None

        # Inicializa detector ORB para capturar mudanças estruturais
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def _compute_hsv_hist(frame: np.ndarray) -> np.ndarray:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist

        def _is_significant_change(current: np.ndarray, reference: np.ndarray, ref_hist: np.ndarray) -> bool:
            """Retorna True se a mudança entre `current` e `reference` for significativa."""
            # Critério 1: Correlação de histograma (1.0 = idêntico, <threshold = mudou)
            curr_hist = _compute_hsv_hist(current)
            hist_corr = cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_CORREL)
            if hist_corr < scene_threshold:
                logger.debug(f"Scene change (histogram corr={hist_corr:.3f} < {scene_threshold})")
                return True, curr_hist

            # Critério 2: ORB matching ratio — poucas correspondencias = cena diferente
            try:
                ref_small = cv2.resize(reference, (320, 180))
                cur_small = cv2.resize(current, (320, 180))
                kp1, des1 = orb.detectAndCompute(ref_small, None)
                kp2, des2 = orb.detectAndCompute(cur_small, None)
                if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                    matches = bf.match(des1, des2)
                    good_matches = [m for m in matches if m.distance < 50]
                    match_ratio = len(good_matches) / max(len(kp1), len(kp2), 1)
                    if match_ratio < (1.0 - scene_threshold):
                        logger.debug(f"Scene change (ORB match_ratio={match_ratio:.3f})")
                        return True, curr_hist
            except Exception as e:
                logger.debug(f"ORB matching skipped: {e}")

            return False, curr_hist

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % hop == 0:
                timestamp = count / fps

                if not detect_change:
                    yield timestamp, frame
                    count += 1
                    continue

                if last_keyframe is None:
                    # Primeiro frame sempre é enviado como keyframe
                    last_keyframe = frame
                    last_hist = _compute_hsv_hist(frame)
                    yield timestamp, frame
                else:
                    changed, new_hist = _is_significant_change(frame, last_keyframe, last_hist)
                    if changed:
                        last_keyframe = frame
                        last_hist = new_hist
                        yield timestamp, frame
                    else:
                        logger.debug(f"Frame at {timestamp:.2f}s skipped (no significant change)")

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
        logger.info("Packing Frames")
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

    async def describe_grids_batch(self, grids: List[Image.Image], prompt: str, headers: dict = None, model: str = "fastvlm") -> List[GridNarrative]:
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
            "headers": headers or {},
            "model": model
        }
        
        from ..bridge import bridge
        results = await bridge.process_image_description(payload, headers or {})
        
        narratives = []
        if isinstance(results, str):
            results = [results]
        elif isinstance(results, dict):
            # Se for um objeto de resposta única, tenta extrair o texto
            text = results.get("choices", [{}])[0].get("message", {}).get("content", str(results))
            results = [text]
            
        for description in results:
            desc_text = description.get("text", "") if isinstance(description, dict) else str(description)
            narratives.append(GridNarrative(
                text=desc_text,
                visual_vibe="dynamic",
                action_summary=desc_text[:100], 
                tokens_consumed=0 
            ))
            
        return narratives


class IVideoAnalysisStrategy(abc.ABC):
    @abc.abstractmethod
    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: VLMAdapter) -> List[VideoSegment]:
        pass


class ObjectInteractionStrategy(IVideoAnalysisStrategy):
    def __init__(self, yolo_model_path='yolo11s.pt'):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_model_path)
            logger.info(f"ObjectInteractionStrategy: loaded YOLO model '{yolo_model_path}'")
        except ImportError:
            self.yolo = None
            logger.warning("ObjectInteractionStrategy: ultralytics not installed, YOLO features disabled.")

    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: VLMAdapter) -> List[VideoSegment]:
        segments = []
        
        if self.yolo is None:
            # Fallback para descrição padrão se YOLO não estiver disponível
            narratives = await vlm.describe_grids_batch(grids, "Describe actions and objects.")
            for tr, nar in zip(time_ranges, narratives):
                segments.append(VideoSegment(tr, nar.text))
            return segments

        spatial_prompt = "Analyze the spatial proximity between the people's faces and objects."
        narratives = await vlm.describe_grids_batch(grids, spatial_prompt)
        
        for tr, nar in zip(time_ranges, narratives):
            segments.append(VideoSegment(tr, nar.text))

        return segments

class EmotionAndAttentionStrategy(IVideoAnalysisStrategy):
    async def analyze_batch(self, grids: List[Image.Image], time_ranges: List[tuple], face_service: FaceService, vlm: VLMAdapter) -> List[VideoSegment]:
        from ..models.emotion import EmotionModel
        # emotion_model = EmotionModel(face_service.detector, backend="cpu")

        # ... lógica de emoção simplificada para usar o VLMAdapter ...
        prompt = "Analyze these images. Deduce body posture and gaze direction."
        narratives = await vlm.describe_grids_batch(grids, prompt)
        
        segments = []
        for i, tr in enumerate(time_ranges):
            desc = narratives[i].text if i < len(narratives) else "Posture unknown."
            segments.append(VideoSegment(tr, desc))

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

    async def execute(self, video_path: str, interval_sec: float = 2.0, frames_per_grid: int = 4, batch_size: int = 10, strategy: str = "default", detect_changes=False, headers: Optional[Dict] = None, model: str = "fastvlm"):
        logger.info(f"Starting multimodal native batch analysis for: {video_path} with strategy: {strategy}")
        
        headers = headers or {}
        video_aggregate = VideoAggregate(
            video_id=str(uuid.uuid4()),
            file_path=video_path,
            metadata={"interval_sec": interval_sec, "strategy": strategy},
            processed_at=datetime.now()
        )

        audio_task = asyncio.create_task(self._process_full_audio(video_path, headers=headers))

        frame_buffer = []
        timestamps = []
        grid_batch = []
        timestamp_batch = [] 
        
        analysis_strategy = None
        if strategy == "interaction":
            analysis_strategy = ObjectInteractionStrategy()
        elif strategy == "emotion":
            analysis_strategy = EmotionAndAttentionStrategy()

        for ts, frame in self.bifurcator.sample_frames(video_path, interval_sec, detect_change=detect_changes):
            frame_buffer.append(frame)
            timestamps.append(ts)
            
            if len(frame_buffer) >= frames_per_grid:
                grid_img = self.packer.pack(frame_buffer)
                grid_batch.append(grid_img)
                timestamp_batch.append((timestamps[0], timestamps[-1]))
                frame_buffer = []
                timestamps = []
                
                if len(grid_batch) >= batch_size:
                    await self._process_batch(video_aggregate, grid_batch, timestamp_batch, analysis_strategy, headers=headers, model=model)
                    grid_batch = []
                    timestamp_batch = []

        if frame_buffer:
            grid_img = self.packer.pack(frame_buffer)
            grid_batch.append(grid_img)
            timestamp_batch.append((timestamps[0], timestamps[-1]))
            
        if grid_batch:
            await self._process_batch(video_aggregate, grid_batch, timestamp_batch, analysis_strategy, headers=headers, model=model)

        logger.info(f"Vision processing complete. Waiting for audio transcription to finish...")
        full_text = await audio_task
        video_aggregate.full_transcription = full_text

        logger.info(f"Multimodal Analysis complete. {len(video_aggregate.segments)} segments batched and created.")
        return video_aggregate

    async def _process_batch(self, aggregate: VideoAggregate, grids: List, time_ranges: List[tuple], strategy: Optional[Any] = None, headers: Optional[Dict] = None, model: str = "fastvlm"):
        logger.info("Analyzing Batch")
        headers = headers or {}
        
        if strategy:
            segments = await strategy.analyze_batch(grids, time_ranges, self.face_service, self.vlm)
            for seg in segments:
                new_seg = VideoSegment(
                    segment_id=str(uuid.uuid4()),
                    start_offset=seg.time_range[0],
                    end_offset=seg.time_range[1],
                    narrative=GridNarrative(
                        text=seg.description,
                        visual_vibe="dynamic",
                        action_summary=seg.description[:100],
                        tokens_consumed=0
                    )
                )
                aggregate.segments.append(new_seg)
        else:
            prompt = "Describe the sequence of actions, environment and people in this video narrative block."
            narratives = await self.vlm.describe_grids_batch(grids, prompt, headers, model)
            
            for i, nar in enumerate(narratives):
                start_ts, end_ts = time_ranges[i]
                segment = VideoSegment(
                    segment_id=str(uuid.uuid4()),
                    start_offset=start_ts,
                    end_offset=end_ts,
                    narrative=nar
                )
                aggregate.segments.append(segment)

    async def _process_full_audio(self, video_path: str, headers: Optional[Dict] = None) -> str:
        """Extrai o .wav do vídeo e manda para o STT via bridge."""
        headers = headers or {}
        try:
            audio_path = self.bifurcator.extract_audio_stream(video_path)
            if not audio_path or not os.path.exists(audio_path):
                return ""

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            from ..bridge import bridge
            payload = {
                "audio": audio_bytes,
                "headers": headers 
            }
            
            logger.info(f"Sending extracted audio to STT engine via bridge...")
            transcription = await bridge.process_stt(payload, headers)
            return transcription if isinstance(transcription, str) else str(transcription)
            
        except Exception as e:
            logger.error(f"Audio transcription failed during video analysis: {e}")
            return ""
        finally:
            if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass


multimodal_video_analysis = MultimodalVideoAnalysisUseCase(OpenCVBifurcator(), NumPyGridPacker(), VLMAdapter())
video_processing_service = multimodal_video_analysis
