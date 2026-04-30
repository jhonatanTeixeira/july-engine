import os
import gc
import logging
import asyncio
from typing import Optional, TYPE_CHECKING, Dict, Any
from io import BytesIO

if TYPE_CHECKING:
    import numpy as np
    import soundfile as sf
    # from chatterbox import Chatterbox

logger = logging.getLogger("JulyEngine.Models.ChatterboxTTS")

class ChatterboxTTS:
    def __init__(self, backend="cpu", model_tag="chatterbox-multilingual"):
        self.backend = backend
        self.model_tag = model_tag
        self.model = None
        self.device = "cuda" if backend == "gpu" else "cpu"
        
        # OFF-LOAD AND IDLE CONFIGS
        self.idle_timeout = 120  # 2 minutos ocioso para offload
        self._idle_task = None
        self._is_offloaded = False

        # 1. A MORDAÇA DE THREADS (Otimização para CPU)
        if self.device == "cpu":
            try:
                import torch
                torch.set_num_threads(2)
                logger.info("ChatterboxTTS: PyTorch CPU threads limitadas a 2 para evitar thrashing.")
            except ImportError:
                pass

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.device == "cpu":
            return 0
        # O Chatterbox Multilingual tem 500M de parâmetros.
        # Em inferência FP16, gasta entre 1.5GB a 1.8GB. Reduzido devido à quantização.
        return 1200

    def _reset_idle_timer(self):
        if self.device == "cpu":
            return
        if self._idle_task is not None:
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_timer())

    async def _idle_timer(self):
        try:
            await asyncio.sleep(self.idle_timeout)
            if self.model is not None and not self._is_offloaded:
                logger.info("ChatterboxTTS: Timeout ocioso atingido. Aplicando Idle Offload (GPU -> CPU)...")
                self.model = self.model.to("cpu")
                self._is_offloaded = True
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except asyncio.CancelledError:
            pass

    def load(self):
        if self.model is None:
            # Assumindo a API oficial em Python do chatterbox/resemble-ai
            from chatterbox import Chatterbox
            try:
                logger.info(f"ChatterboxTTS: Loading model {self.model_tag} on {self.device}")
                # Carrega a versão multilingue otimizada
                self.model = Chatterbox.from_pretrained("resemble-ai/chatterbox-multilingual")
                
                # QUANTIZAÇÃO: Usar FP16 para reduzir uso de memória
                if self.device == "cuda":
                    try:
                        self.model = self.model.half()
                        logger.info("ChatterboxTTS: Modelo quantizado para FP16.")
                    except Exception as e:
                        logger.warning(f"ChatterboxTTS: Falha ao aplicar FP16: {e}")
                        
                self.model = self.model.to(self.device)
                self._is_offloaded = False
                logger.info("ChatterboxTTS loaded successfully.")
            except Exception as e:
                logger.error(f"ChatterboxTTS: Failed to load: {e}")
                raise e

    def is_loaded(self):
        return self.model is not None

    def clear(self):
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
            
        if self.model is not None:
            logger.info("ChatterboxTTS: Unloading model")
            del self.model
            self.model = None
            
            # OTIMIZAÇÃO: Limpeza profunda e forçada da VRAM do PyTorch
            gc.collect()
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except ImportError:
                    pass

    # Novos parâmetros nativos do Chatterbox: 'exaggeration' (emoção) e 'temperature'
    async def run(self, 
                  text: str, 
                  voice_id: str, 
                  lang_code: str, 
                  stream: bool = False, 
                  semitones: float = 0.0,
                  exaggeration: float = 0.8, 
                  temperature: float = 0.5):
        
        from ..services.voice_service import voice_service
        
        # 2. ZERO-SHOT CLONING: O Chatterbox precisa do arquivo .wav do speaker
        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            logger.error(f"ChatterboxTTS: Voice {voice_id} not found and no fallback available.")
            raise ValueError(f"Voice {voice_id} not found")
            
        voice_path, _ = voice_res

        if self.model is None:
            self.load()
        elif self._is_offloaded and self.device == "cuda":
            logger.info("ChatterboxTTS: Acordando modelo. Movendo da CPU para GPU (Memory Offload)...")
            self.model = self.model.to(self.device)
            self._is_offloaded = False
            
        self._reset_idle_timer()
            
        try:
            # Inicializa o PitchShift (Sem criar overhead no loop)
            board = None
            if semitones != 0.0:
                try:
                    from pedalboard import Pedalboard, PitchShift
                    board = Pedalboard([PitchShift(semitones=semitones)])
                    logger.info(f"ChatterboxTTS: Pedalboard ativado com PitchShift de {semitones} semitons.")
                except ImportError:
                    logger.warning("ChatterboxTTS: pedalboard não instalado. Ignorando alteração de pitch.")

            # MODO STREAMING (Ideal usar RAW PCM, mas mantive a estrutura segura)
            if stream:
                async def audio_streamer():
                    import numpy as np
                    
                    # O Chatterbox tem suporte a gerador de streaming em chunks
                    generator = self.model.synthesize_stream(
                        text, 
                        language=lang_code, 
                        reference_audio=voice_path,
                        exaggeration=exaggeration,
                        temperature=temperature
                    )
                    
                    def get_next_chunk():
                        try:
                            return next(generator)
                        except StopIteration:
                            return None

                    while True:
                        try:
                            # Puxa o pedaço de áudio gerado
                            audio_chunk = await asyncio.to_thread(get_next_chunk)
                            if audio_chunk is None:
                                break
                            
                            # Opcional: Garante que o áudio é um numpy array flat no CPU
                            if hasattr(audio_chunk, 'cpu'):
                                audio_chunk = audio_chunk.cpu().numpy()
                            
                            if board:
                                # O Chatterbox gera em 24000Hz por defeito
                                audio_chunk = board(audio_chunk, 24000)
                                
                            # Convert to WAV format in memory
                            import soundfile as sf
                            buffer = BytesIO()
                            sf.write(buffer, audio_chunk, 24000, format='WAV')
                            yield buffer.getvalue()
                            
                        except Exception as e:
                            logger.error(f"ChatterboxTTS: Error in streamer: {e}")
                            break
                return audio_streamer()

            # MODO SYNC (Gera a frase toda de uma vez)
            def generate_sync():
                out = self.model.synthesize(
                    text, 
                    language=lang_code, 
                    reference_audio=voice_path,
                    exaggeration=exaggeration,
                    temperature=temperature
                )
                if hasattr(out, 'cpu'):
                    return out.cpu().numpy()
                return out

            final_audio = await asyncio.to_thread(generate_sync)
            
            if board:
                final_audio = board(final_audio, 24000)
            
            # Apenas no modo Sync convertemos para um ficheiro WAV bonitinho
            import soundfile as sf
            buffer = BytesIO()
            sf.write(buffer, final_audio, 24000, format='WAV')
            
            logger.info(f"Engine ChatterboxTTS executed successfully on {self.backend}")
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"ChatterboxTTS execution failed: {e}")
            raise e