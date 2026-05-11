import asyncio
import re
import logging
from typing import Any, Dict, Optional, Union, AsyncGenerator

from ..models.base_model import BaseModel

logger = logging.getLogger("JulyEngine.Adapter.TTSAdapter")

# Maps model alias prefixes → tts_engine key
_ALIAS_ENGINE_MAP = [
    ("kokoro",       "kokoro"),
    ("chatterbox",   "chatterbox"),
    ("qwen3-tts",    "qwen3"),
    ("qwen3_tts",    "qwen3"),
    ("xtts",         "xtts2"),
    ("piper",        "piper"),
]

_MARKDOWN_RE = re.compile(r'[*_`#~\[\]()\\<>+=\-|{}]')
_SENTENCE_RE = re.compile(r'[.\n]+')


class TTSAdapter(BaseModel):
    """
    Aggregates all TTS engines with text preprocessing, streaming routing,
    and config injection from the persistence backend.

    Engine field: "tts"

    Sub-engine selection (in order):
      1. meta["tts_engine"] field
      2. alias prefix matching against _ALIAS_ENGINE_MAP
      3. Falls back to "api"
    """

    def __init__(self, backend: str = "cpu", model_meta: Optional[dict] = None):
        super().__init__(backend, model_meta)
        self._tts_model = None

    # ------------------------------------------------------------------
    # Sub-engine resolution
    # ------------------------------------------------------------------

    def _detect_engine(self) -> str:
        explicit = self.meta.get("tts_engine", "")
        if explicit:
            return explicit
        alias = (self.meta.get("alias") or self.meta.get("model", "")).lower()
        for prefix, engine in _ALIAS_ENGINE_MAP:
            if alias.startswith(prefix):
                return engine
        return "api"

    def _get_tts_model(self):
        if self._tts_model is not None:
            return self._tts_model
        engine = self._detect_engine()
        if engine == "kokoro":
            from ..models.tts_kokoro import KokoroTTSModel
            self._tts_model = KokoroTTSModel(backend=self.backend, model_meta=self.meta)
        elif engine == "chatterbox":
            from ..models.tts_chatterbox import ChatterboxTTSModel
            self._tts_model = ChatterboxTTSModel(backend=self.backend, model_meta=self.meta)
        elif engine == "xtts2":
            from ..models.tts_xtts2 import XTTS2Model
            self._tts_model = XTTS2Model(backend=self.backend, model_meta=self.meta)
        elif engine == "piper":
            from ..models.tts_piper import PiperModel
            self._tts_model = PiperModel(backend=self.backend, model_meta=self.meta)
        elif engine == "qwen3":
            from ..models.tts_qwen3 import FasterQwen3TTSModel
            self._tts_model = FasterQwen3TTSModel(backend=self.backend, model_meta=self.meta)
        # "api" — no local model instance
        return self._tts_model

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        m = self._get_tts_model()
        if m and hasattr(m, "get_required_vram"):
            result = m.get_required_vram(payload)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return 0

    def load(self, n_ctx=None, num_layers=None):
        m = self._get_tts_model()
        if m:
            m.load()

    def is_loaded(self) -> bool:
        return self._tts_model is not None and self._tts_model.is_loaded()

    def unload(self, model_name=None):
        if self._tts_model:
            self._tts_model.unload()

    async def run(self, payload: Dict[str, Any]):
        raw_text = payload.get("input") or payload.get("text", "")
        clean_text = _MARKDOWN_RE.sub("", raw_text)

        stream: bool = payload.get("stream", False)
        headers: dict = payload.get("headers", {})

        config = self._load_tts_config()
        voice_id = payload.get("voice") or config.get("voice", "af_heart")
        language = payload.get("language") or config.get("language", "a")
        temperature = payload.get("temperature") or config.get("temperature") or 0.7
        semitones = payload.get("semitones") or config.get("semitones") or 0.0

        if config:
            headers.setdefault("x-base-url", config.get("base_url"))
            headers.setdefault("x-api-key", config.get("api_key"))
            if config.get("api_key"):
                headers.setdefault("authorization", f"Bearer {config['api_key']}")

        engine = self._detect_engine()
        tts_payload = {
            **payload,
            "input": clean_text,
            "voice": voice_id,
            "language": language,
            "temperature": temperature,
            "semitones": semitones,
            "headers": headers,
        }

        if stream:
            return await self._dispatch_stream(engine, clean_text, tts_payload)
        return await self._dispatch_sync(engine, tts_payload)

    # ------------------------------------------------------------------
    # Streaming dispatch
    # ------------------------------------------------------------------

    async def _dispatch_stream(self, engine: str, clean_text: str, payload: dict):
        # Native streaming engines — yield chunks directly
        if engine in ("kokoro", "chatterbox"):
            model = self._get_tts_model()
            result = await model.run({**payload, "stream": True})
            return result

        if engine == "qwen3":
            model = self._get_tts_model()
            result = model.run({**payload, "stream": True})
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # Fallback chunking: XTTS2, Piper, API — split text into sentences
        sentences = [s.strip() + "." for s in _SENTENCE_RE.split(clean_text) if s.strip()]

        adapter = self

        async def sentence_streamer() -> AsyncGenerator[bytes, None]:
            for sentence in sentences:
                chunk = await adapter._synth_sentence(engine, sentence, payload)
                if chunk:
                    yield chunk

        return sentence_streamer()

    # ------------------------------------------------------------------
    # Non-streaming dispatch
    # ------------------------------------------------------------------

    async def _dispatch_sync(self, engine: str, payload: dict):
        if engine == "api":
            from ..services.llm_api import llm_api
            return await llm_api.dispatch("tts", {**payload, "stream": False})

        model = self._get_tts_model()
        if not model:
            return None

        result = model.run({**payload, "stream": False})
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _synth_sentence(self, engine: str, sentence: str, payload: dict) -> Optional[bytes]:
        p = {**payload, "input": sentence, "stream": False}
        if engine == "api":
            from ..services.llm_api import llm_api
            return await llm_api.dispatch("tts", p)
        model = self._get_tts_model()
        if not model:
            return None
        result = model.run(p)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _load_tts_config(self) -> dict:
        try:
            from ..services.models_service import model_service
            return model_service.backend.get_setting("TTS") or {}
        except Exception:
            return {}
