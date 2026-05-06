from typing import Union
import os
import io
import logging
from typing import Any, Dict, Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger("JulyEngine.Models.FasterQwen3TTS")

class FasterQwen3TTS:
    """
    Faster Qwen3-TTS model para síntese de alta velocidade.
    Otimizado para máxima economia de VRAM em GPUs consumer-grade.
    """
    def __init__(self, backend="gpu", model_size="12Hz-0.6B-Base"):
        self.backend = backend
        self.device = "cpu" # Default until load()
        self.model = None
        self.model_size = model_size # "0.6B" (Recomendado para restrição de VRAM) ou "1.7B"
        
        self._is_offloaded = False

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM estimada. O 0.6B em bf16 consome ~2.6GB + overhead."""
        if self.backend == "cpu":
            return 0

        return 3000 if "0.6B" in self.model_size else 6500

    def load(self):
        if self.model is None:
            import torch
            from faster_qwen3_tts import FasterQwen3TTS
            
            self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
            
            try:
                logger.info(f"FasterQwen3TTS: Loading {self.model_size} model on {self.device}")
                
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                
                self.model = FasterQwen3TTS.from_pretrained(
                    f"Qwen/Qwen3-TTS-{self.model_size}",
                    device=self.device,
                    dtype=dtype
                )
                
                logger.info("FasterQwen3TTS loaded successfully.")
            except Exception as e:
                logger.error(f"FasterQwen3TTS: Failed to load: {e}")
                raise e

    def run(self, text: str, voice_id: str, language: str, temperature: float = 0.7, stream: bool = False) -> Union[bytes, 'Generator[bytes, None, None]']:
        from ..services.voice_service import voice_service
        from typing import Generator
        
        voice_res = voice_service.get_voice_path(voice_id)
        
        if not voice_res:
            logger.error(f"FasterQwen3TTS: Voice {voice_id} not found and no fallback available.")
            raise ValueError(f"Voice {voice_id} not found")
            
        voice_path, voice_lang = voice_res
        target_lang = language or voice_lang

        if self.model is None:
            self.load()
            
        # Limpeza de texto focada na estabilidade da arquitetura do Qwen3
        text = text.replace('"', '').replace("-", " ").strip()

        if not text.endswith((".", "!", "?")):
            text += "."
            
        logger.info(f"FasterQwen3TTS: Synthesizing using speaker {voice_path} (Stream: {stream})")

        if stream:
            def streamer() -> Generator[bytes, None, None]:
                try:
                    for wav_chunk, sr, timing in self.model.generate_voice_clone_streaming(
                        text=text,
                        language=target_lang,
                        ref_audio=voice_path,
                        temperature=temperature,
                        xvec_only=True
                    ):
                        buffer = io.BytesIO()
                        sf.write(buffer, np.array(wav_chunk), sr, format='WAV')
                        yield buffer.getvalue()
                except Exception as e:
                    logger.error(f"FasterQwen3TTS Stream Error: {e}")

            return streamer()

        try:
            wav_arrays, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=target_lang,
                ref_audio=voice_path,
                temperature=temperature,
                xvec_only=True
            )
            
            # Conversão in-memory concatenando todos os arrays
            wav_array = np.concatenate(wav_arrays) if isinstance(wav_arrays, list) else wav_arrays
            buffer = io.BytesIO()
            sf.write(buffer, wav_array, sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            # OTIMIZAÇÃO VRAM 4: Limpeza síncrona pós-inferência
            del wav_array
            del wav_arrays
                
            logger.info(f"Engine FasterQwen3TTS executed successfully on {self.backend}")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"FasterQwen3TTS: Execution failed: {e}")
            raise e

    def is_loaded(self):
        return self.model is not None

    def unload(self):
        """Libera os recursos e zera a VRAM retida."""
            
        if self.model:
            del self.model
            self.model = None

        from ..resource_manager import resource_manager
        resource_manager.clear_memory()

        logger.info("FasterQwen3TTS: Model unloaded.")