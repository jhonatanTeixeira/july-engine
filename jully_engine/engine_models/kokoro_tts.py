import os
import gc
import logging
import asyncio
from typing import Optional, TYPE_CHECKING, Dict, Any
from io import BytesIO

if TYPE_CHECKING:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

logger = logging.getLogger("JulyEngine.Models.KokoroTTS")

class KokoroTTS:
    def __init__(self, backend="cpu", model_tag="kokoro"):
        self.backend = backend
        self.model_tag = model_tag
        self.pipeline = None
        self.device = "cuda" if backend == "gpu" else "cpu"
        self.lang_code = None

        if self.device == "cpu":
            try:
                import torch
                torch.set_num_threads(2)
                logger.info("KokoroTTS: PyTorch CPU threads limitadas a 2 para evitar thrashing.")
            except ImportError:
                pass

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.device == "cpu":
            return 0
        return 500

    def load(self, lang_code='a'):
        if self.pipeline is None or lang_code != self.lang_code:
            from kokoro import KPipeline
            try:
                logger.info(f"KokoroTTS: Loading model on {self.device}")
                self.pipeline = KPipeline(lang_code=lang_code, device=self.device)
                self.lang_code = lang_code
                logger.info("KokoroTTS loaded successfully.")
            except Exception as e:
                logger.error(f"KokoroTTS: Failed to load: {e}")
                raise e

    def is_loaded(self):
        return self.pipeline is not None

    def clear(self):
        if self.pipeline is not None:
            logger.info("KokoroTTS: Unloading model")
            del self.pipeline
            self.pipeline = None
            
            # OTIMIZAÇÃO: Limpeza profunda (RAM e VRAM)
            gc.collect()
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except ImportError:
                    pass

    async def run(self, text: str, voice_id: str, lang_code: str, stream: bool = False, semitones: float = 0.0):
        self.load(lang_code)
            
        try:
            generator = self.pipeline(text, voice=voice_id, speed=1.0, split_pattern=r'\n+')
            
            board = None
            if semitones != 0.0:
                try:
                    from pedalboard import Pedalboard, PitchShift
                    board = Pedalboard([PitchShift(semitones=semitones)])
                    logger.info(f"KokoroTTS: Pedalboard ativado com PitchShift de {semitones} semitons.")
                except ImportError:
                    logger.warning("KokoroTTS: pedalboard não instalado. Ignorando alteração de pitch.")
            
            if stream:
                async def audio_streamer():
                    # OTIMIZAÇÃO: Import fora do loop `while`
                    import soundfile as sf
                    import numpy as np
                    
                    def get_next_chunk():
                        try:
                            return next(generator)
                        except StopIteration:
                            return None

                    while True:
                        try:
                            # O generator do Kokoro devolve (graphemes, phonemes, audio)
                            res = await asyncio.to_thread(get_next_chunk)
                            if res is None:
                                break
                            
                            _, _, audio = res
                            
                            if board:
                                audio = board(audio, 24000)
                                
                            buffer = BytesIO()
                            sf.write(buffer, audio, 24000, format='WAV')
                            
                            yield buffer.getvalue()
                            
                        except Exception as e:
                            logger.error(f"KokoroTTS: Error in streamer: {e}")
                            break
                return audio_streamer()

            def generate_all_sync():
                import numpy as np
                return np.concatenate([audio for _, _, audio in generator])

            final_audio = await asyncio.to_thread(generate_all_sync)
            
            if board:
                final_audio = board(final_audio, 24000)
            
            import soundfile as sf
            buffer = BytesIO()
            sf.write(buffer, final_audio, 24000, format='WAV')
            
            logger.info(f"Engine KokoroTTS executed successfully on {self.backend}")
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"KokoroTTS execution failed: {e}")
            raise e