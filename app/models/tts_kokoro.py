import asyncio
import gc
import logging
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, Optional, Union

from .base_model import BaseModel

logger = logging.getLogger("JulyEngine.Models.KokoroTTS")


class KokoroTTSModel(BaseModel):
    """
    Kokoro TTS with two output forms:
      - stream=False → bytes (full WAV)
      - stream=True  → AsyncGenerator[bytes] (WAV chunks via async generator)
    """

    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self.device = "cuda" if backend == "gpu" else "cpu"
        self._pipeline = None
        self._lang_code = None

        if self.device == "cpu":
            try:
                import torch
                torch.set_num_threads(2)
            except ImportError:
                pass

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        return 0 if self.backend == "cpu" else 500

    def load(self, n_ctx=None, num_layers=None):
        # Pipeline loads per language code; lazy-loaded in run()
        pass

    def _ensure_pipeline(self, lang_code: str):
        if self._pipeline is None or lang_code != self._lang_code:
            from kokoro import KPipeline
            logger.info(f"KokoroTTS: Loading pipeline for lang={lang_code} on {self.device}")
            self._pipeline = KPipeline(lang_code=lang_code, device=self.device)
            self._lang_code = lang_code
            logger.info("KokoroTTS loaded.")

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def unload(self, model_name=None):
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        gc.collect()
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except ImportError:
                pass

    async def run(self, payload: Dict[str, Any], **kwargs) -> Union[bytes, AsyncGenerator[bytes, None]]:
        text = payload.get("input") or payload.get("text", "")
        voice_id = payload.get("voice", "af_heart")
        lang_code = payload.get("language") or "a" 
        semitones = payload.get("semitones")
        if semitones is None:
            semitones = 0.0
        else:
            semitones = float(semitones)
        
        stream = payload.get("stream", False)

        logger.debug(f"KokoroTTS: run(lang={lang_code}, voice={voice_id}, stream={stream}, semitones={semitones})")
        self._ensure_pipeline(lang_code)

        generator = self._pipeline(text, voice=voice_id, speed=1.0, split_pattern=r"\n+")
        board = _make_pitch_board(semitones)

        if stream:
            return self._stream(generator, board)

        return await asyncio.to_thread(self._collect_all, generator, board)

    # ------------------------------------------------------------------
    # Streaming form: async generator (native yield)
    # ------------------------------------------------------------------

    async def _stream(self, generator, board) -> AsyncGenerator[bytes, None]:
        import soundfile as sf
        import numpy as np

        def next_chunk():
            try:
                return next(generator)
            except StopIteration:
                return None

        while True:
            res = await asyncio.to_thread(next_chunk)
            if res is None:
                break
            _, _, audio = res
            if board:
                audio = board(audio, 24000)
            buf = BytesIO()
            sf.write(buf, audio, 24000, format="WAV")
            yield buf.getvalue()

    # ------------------------------------------------------------------
    # Batch form: collect everything then return bytes
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_all(generator, board) -> bytes:
        import numpy as np
        import soundfile as sf

        chunks = [audio for _, _, audio in generator]
        if not chunks:
            logger.warning("KokoroTTS: No audio chunks were generated.")
            return b""

        final = np.concatenate(chunks)
        if board:
            final = board(final, 24000)
        buf = BytesIO()
        sf.write(buf, final, 24000, format="WAV")
        return buf.getvalue()


def _make_pitch_board(semitones: float):
    if semitones == 0.0:
        return None
    try:
        from pedalboard import Pedalboard, PitchShift
        return Pedalboard([PitchShift(semitones=semitones)])
    except ImportError:
        logger.warning("KokoroTTS: pedalboard not installed — pitch shift skipped")
        return None
