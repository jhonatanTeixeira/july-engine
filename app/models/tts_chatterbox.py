import asyncio
import gc
import logging
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, Optional, Union

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.ChatterboxTTS")

IDLE_TIMEOUT = 120


class ChatterboxTTSModel(BaseModel):
    """
    Chatterbox TTS (zero-shot voice cloning) with two output forms:
      - stream=False → bytes (full WAV)
      - stream=True  → AsyncGenerator[bytes] (WAV chunks via async generator)

    Supports idle GPU→CPU offload after IDLE_TIMEOUT seconds.
    """

    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.device = "cuda" if backend == "gpu" else "cpu"
        self._model = None
        self._idle_task = None
        self._is_offloaded = False

        if self.device == "cpu":
            try:
                import torch
                torch.set_num_threads(2)
            except ImportError:
                pass

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 1200

    def load(self, n_ctx=None, num_layers=None):
        if self._model is not None:
            return
        from chatterbox import Chatterbox

        logger.info(f"ChatterboxTTS: Loading on {self.device}")
        self._model = Chatterbox.from_pretrained("resemble-ai/chatterbox-multilingual")

        if self.device == "cuda":
            try:
                self._model = self._model.half()
            except Exception as e:
                logger.warning(f"ChatterboxTTS: FP16 failed: {e}")

        self._model = self._model.to(self.device)
        self._is_offloaded = False
        logger.info("ChatterboxTTS loaded.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self, model_name=None):
        self._cancel_idle()
        if self._model:
            del self._model
            self._model = None
        gc.collect()
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except ImportError:
                pass
        logger.info("ChatterboxTTS unloaded.")

    async def run(self, payload: Dict[str, Any], **kwargs) -> Union[bytes, AsyncGenerator[bytes, None]]:
        from ..services.voice_service import voice_service

        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "default")
        lang_code = payload.get("language", "en")
        stream = payload.get("stream", False)
        semitones = payload.get("semitones", 0.0)
        exaggeration = payload.get("exaggeration", 0.8)
        temperature = payload.get("temperature", 0.5)

        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            raise ValueError(f"ChatterboxTTS: Voice '{voice_id}' not found")
        voice_path, _ = voice_res

        if self._model is None:
            self.load()
        elif self._is_offloaded and self.device == "cuda":
            logger.info("ChatterboxTTS: Waking up from idle offload (CPU → GPU)")
            self._model = self._model.to(self.device)
            self._is_offloaded = False

        self._reset_idle()
        board = _make_pitch_board(semitones)

        if stream:
            return self._stream(text, voice_path, lang_code, exaggeration, temperature, board)

        return await self._generate_full(text, voice_path, lang_code, exaggeration, temperature, board)

    # ------------------------------------------------------------------
    # Streaming form: async generator (native yield)
    # ------------------------------------------------------------------

    async def _stream(self, text, voice_path, lang_code, exaggeration, temperature, board) -> AsyncGenerator[bytes, None]:
        import numpy as np
        import soundfile as sf

        sync_gen = self._model.synthesize_stream(
            text, language=lang_code, reference_audio=voice_path,
            exaggeration=exaggeration, temperature=temperature,
        )

        def next_chunk():
            try:
                return next(sync_gen)
            except StopIteration:
                return None

        while True:
            chunk = await asyncio.to_thread(next_chunk)
            if chunk is None:
                break
            if hasattr(chunk, "cpu"):
                chunk = chunk.cpu().numpy()
            if board:
                chunk = board(chunk, 24000)
            buf = BytesIO()
            sf.write(buf, chunk, 24000, format="WAV")
            yield buf.getvalue()

    # ------------------------------------------------------------------
    # Batch form
    # ------------------------------------------------------------------

    async def _generate_full(self, text, voice_path, lang_code, exaggeration, temperature, board) -> bytes:
        import soundfile as sf

        def synthesize():
            out = self._model.synthesize(
                text, language=lang_code, reference_audio=voice_path,
                exaggeration=exaggeration, temperature=temperature,
            )
            return out.cpu().numpy() if hasattr(out, "cpu") else out

        audio = await asyncio.to_thread(synthesize)
        if board:
            audio = board(audio, 24000)
        buf = BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        return buf.getvalue()

    # ------------------------------------------------------------------

    def _reset_idle(self):
        if self.device == "cpu":
            return
        self._cancel_idle()
        self._idle_task = asyncio.create_task(self._idle_timer())

    def _cancel_idle(self):
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_timer(self):
        try:
            await asyncio.sleep(IDLE_TIMEOUT)
            if self._model and not self._is_offloaded:
                logger.info("ChatterboxTTS: Idle timeout — offloading GPU → CPU")
                self._model = self._model.to("cpu")
                self._is_offloaded = True
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except asyncio.CancelledError:
            pass


def _make_pitch_board(semitones: float):
    if semitones == 0.0:
        return None
    try:
        from pedalboard import Pedalboard, PitchShift
        return Pedalboard([PitchShift(semitones=semitones)])
    except ImportError:
        logger.warning("ChatterboxTTS: pedalboard not installed — pitch shift skipped")
        return None
