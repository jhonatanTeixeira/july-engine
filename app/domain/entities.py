from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass(frozen=True)
class PoseMetrics:
    """Métricas estruturais de movimento capturadas em frames"""
    skeleton_map: Dict[str, List[float]]
    confidence_score: float
    movement_delta: float


@dataclass(frozen=True)
class EmotionProfile:
    """Perfil sentimental extraído via microexpressões ou áudio"""
    dominant_emotion: str
    intensity: float
    valence: float


@dataclass(frozen=True)
class FrameEvent:
    """Um evento singular detectado em um timestamp específico"""
    timestamp: float
    pose: Optional[PoseMetrics] = None
    emotion: Optional[EmotionProfile] = None
    detected_objects: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GridNarrative:
    """A narrativa rica gerada pelo VLM a partir de um Grid de frames"""
    text: str
    visual_vibe: str
    action_summary: str
    tokens_consumed: int


@dataclass
class VideoSegment:
    """Entidade que representa um trecho lógico do vídeo"""
    segment_id: str
    start_offset: float
    end_offset: float
    narrative: Optional[GridNarrative] = None
    events: List[FrameEvent] = field(default_factory=list)


@dataclass
class VideoAggregate:
    """Aggregate Root para o domínio de Vídeo"""
    video_id: str
    file_path: str
    metadata: Dict
    processed_at: datetime
    segments: List[VideoSegment] = field(default_factory=list)
    full_transcription: Optional[str] = None