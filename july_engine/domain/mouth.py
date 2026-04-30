from __future__ import annotations
import logging
import os
import json
import re
import inspect
from typing import Any, Dict, Optional, Union, AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine_models.replicate_api import Replicate
    from ..engine_models.xtts2 import XTTS2
    from ..engine_models.piper import Piper
    from ..engine_models.llm_api import LLMApi
    from ..engine_models.kokoro_tts import KokoroTTS
    from ..engine_models.chatterbox import ChatterboxTTS
    from ..engine_models.qwen3_tts import FasterQwen3TTS

from ..persistence import get_backend

logger = logging.getLogger("JulyEngine.Domain.Mouth")

class Mouth:
    """
    Handles Text-to-Speech (TTS) logic and voice resolution.
    Strategies: XTTS2 (cpu, gpu), Piper (cpu, gpu), KokoroTTS (cpu, gpu), LLMApi (api).
    """
    def __init__(self, backend: str, model_tag: str):
        self.backend = backend
        self.model_tag = model_tag
        self._strategy = self._get_strategy()
        self.voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "voices"))
        self.persistence_backend = get_backend()

    def _get_strategy(self):
        if self.backend == "api":
            if self.model_tag.startswith('replicate/'):
                from ..engine_models.replicate_api import Replicate
                return Replicate()
            
            from ..engine_models.llm_api import LLMApi
            return LLMApi(backend=self.backend)
        elif self.model_tag == "xtts":
            from ..engine_models.xtts2 import XTTS2
            return XTTS2(backend=self.backend)
        elif self.model_tag == "piper":
            from ..engine_models.piper import Piper
            return Piper(backend=self.backend)
        elif self.model_tag.startswith("kokoro"):
            from ..engine_models.kokoro_tts import KokoroTTS
            return KokoroTTS(backend=self.backend, model_tag=self.model_tag)
        elif self.model_tag.startswith("chatterbox"):
            from ..engine_models.chatterbox_tts import ChatterboxTTS
            return ChatterboxTTS(backend=self.backend, model_tag=self.model_tag)
        elif self.model_tag == "qwen3-tts":
            from ..engine_models.qwen3_tts import FasterQwen3TTS
            return FasterQwen3TTS(backend=self.backend)
        else:
            raise ValueError(f"Mouth: Unsupported backend/model combination: {self.backend}/{self.model_tag}")

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Delega a estimativa de VRAM para a estratégia atual."""
        if hasattr(self._strategy, "get_required_vram"):
            res = self._strategy.get_required_vram(payload)
            if inspect.iscoroutine(res):
                return await res
            return res
        return 0

    async def speak(self, payload: Dict[str, Any]) -> Union[bytes, AsyncGenerator[bytes, None]]:
        from ..engine_models.replicate_api import Replicate
        from ..engine_models.xtts2 import XTTS2
        from ..engine_models.piper import Piper
        from ..engine_models.llm_api import LLMApi
        from ..engine_models.kokoro_tts import KokoroTTS
        from ..engine_models.qwen3_tts import FasterQwen3TTS
        
        # 1. Extração e Limpeza de Texto
        raw_text = payload.get("input", payload.get("text", ""))
        clean_text = re.sub(r'[*_`#~\[\]()\\<>+=\-|{}]', '', raw_text)
        
        headers: dict = payload.get("headers", {})
        stream: bool = payload.get("stream", False)
        
        from ..persistence import get_backend
        config = get_backend().get_setting("TTS") or {}
        
        voice_id = payload.get("voice", None) or config.get('voice', 'af_heart')
        language = payload.get("language", None) or config.get('language', 'a')
        temperature = payload.get("temperature", None) or config.get('temperature', None) or 0.7
        semitones = payload.get("semitones", None) or config.get('semitones', None) or 0.0

        if config:
            headers.setdefault('x-base-url', config.get('base_url', None))
            headers.setdefault("x-api-key", config.get('api_key', None))
            headers.setdefault("authorization", f"Bearer {config.get('api_key', None)}")

        # ==========================================
        # 2. LÓGICA INTELIGENTE DE STREAMING
        # ==========================================
        if stream:
            
            # --- Rota A: Modelos com Streaming NATIVO (Zero latência) ---
            if isinstance(self._strategy, KokoroTTS):
                # O Kokoro foi construído para devolver os bytes quase instantaneamente
                # Passamos o texto inteiro e deixamos a engine lidar com os buffers de yield
                return await self._strategy.run(
                    clean_text, voice_id, language, stream=True, semitones=semitones
                )
            
            # (Adicione aqui o Chatterbox se for usar o generator dele)
            # elif isinstance(self._strategy, ChatterboxTTS):
            #     return await self._strategy.run(clean_text, voice_id, language, stream=True, ...)

            # --- Rota B: Fallback Chunking (Para XTTS2, Piper e APIs engessadas) ---
            # Aqui mantemos a sua lógica brilhante de poupar RAM cortando a frase!
            sentences = [s.strip() + "." for s in re.split(r'[.\n]+', clean_text) if s.strip()]
            logger.debug(f"Mouth: Chunking text into {len(sentences)} sentences.")
            
            async def sentence_streamer():
                for sentence in sentences:
                    logger.info(f"Mouth: Streaming fallback chunk: {sentence[:50]}...")
                    
                    audio_chunk = None
                    if isinstance(self._strategy, (LLMApi, Replicate)):
                        audio_chunk = await self._strategy.run_tts(self.model_tag, sentence, voice_id, headers=headers)
                    elif isinstance(self._strategy, XTTS2):
                        audio_chunk = self._strategy.run(sentence, voice_id, language, temperature=temperature)
                    elif isinstance(self._strategy, Piper):
                        audio_chunk = self._strategy.run(sentence, voice_id)
                    elif isinstance(self._strategy, FasterQwen3TTS):
                        audio_chunk = self._strategy.run(sentence, voice_id, language, temperature=temperature)

                    if audio_chunk:
                        if inspect.iscoroutine(audio_chunk):
                            audio_chunk = await audio_chunk
                        yield audio_chunk
            
            return sentence_streamer()

        # ==========================================
        # 3. Lógica Não-Streaming (Frase inteira de uma vez)
        # ==========================================
        if isinstance(self._strategy, (LLMApi, Replicate)):
            audio_content = self._strategy.run_tts(self.model_tag, clean_text, voice_id, headers=headers, **payload)
            if inspect.iscoroutine(audio_content):
                audio_content = await audio_content
            return audio_content

        if isinstance(self._strategy, XTTS2):
            return self._strategy.run(clean_text, voice_id, language, temperature=temperature)

        elif isinstance(self._strategy, Piper):
            return self._strategy.run(clean_text, voice_id)

        elif isinstance(self._strategy, KokoroTTS):
            return await self._strategy.run(clean_text, voice_id, language, stream=False, semitones=semitones)

        elif isinstance(self._strategy, FasterQwen3TTS):
            return self._strategy.run(clean_text, voice_id, language, temperature=temperature)

        return None

    def is_loaded(self):
        return hasattr(self._strategy, "is_loaded") and self._strategy.is_loaded()

    def load(self):
        if hasattr(self._strategy, "load"):
            self._strategy.load()

    def unload(self):
        """Libera os recursos da estratégia (XTTS2, Piper, etc)."""
        if hasattr(self._strategy, "unload"):
            self._strategy.unload()
            logger.info(f"Mouth: Strategy {self.model_tag} unloaded.")
        elif hasattr(self._strategy, "clear"):
            self._strategy.clear()
            logger.info(f"Mouth: Strategy {self.model_tag} cleared.")
