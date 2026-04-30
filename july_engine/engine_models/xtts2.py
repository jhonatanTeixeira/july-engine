import os
import io
import logging
from typing import Any, Dict, Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger("JulyEngine.Models.XTTS2")

class XTTS2:
    """
    XTTS v2 model for high-quality TTS.
    Backends: cpu, gpu.
    """
    def __init__(self, backend="gpu"):
        self.backend = backend
        self.device = "cpu" # Default until load()
        self.model = None
        
        # OFF-LOAD AND IDLE CONFIGS
        self.idle_timeout = 120  # 2 minutos ocioso para offload
        self._idle_timer = None
        self._is_offloaded = False

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM para o XTTS2."""
        if self.backend == "cpu":
            return 0
        return 2500

    def _reset_idle_timer(self):
        if self.device == "cpu":
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        import threading
        self._idle_timer = threading.Timer(self.idle_timeout, self._perform_idle_offload)
        self._idle_timer.start()

    def _perform_idle_offload(self):
        if self.model is not None and not self._is_offloaded:
            logger.info("XTTS2: Timeout ocioso atingido. Aplicando Idle Offload (GPU -> CPU)...")
            self.model = self.model.to("cpu")
            self._is_offloaded = True
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load(self):
        if self.model is None:
            import torch
            from TTS.api import TTS
            
            self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
            
            try:
                logger.info(f"XTTS2: Loading model on {self.device}")
                # Otimização: Adicionado carregamento padrão otimizado do XTTS
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                self.model = self.model.to(self.device)
                self._is_offloaded = False
                logger.info("XTTS2 loaded successfully.")
            except Exception as e:
                logger.error(f"XTTS2: Failed to load: {e}")
                raise e

    def run(self, text: str, voice_id: str, language: str, temperature: float = 0.7) -> bytes:
        from ..services.voice_service import voice_service
        
        voice_res = voice_service.get_voice_path(voice_id)
        if not voice_res:
            logger.error(f"XTTS2: Voice {voice_id} not found and no fallback available.")
            raise ValueError(f"Voice {voice_id} not found")
            
        voice_path, voice_lang = voice_res
        
        # Correção do Bug: Usando a variável target_lang corretamente
        target_lang = language or voice_lang

        if self.model is None:
            self.load()
        elif self._is_offloaded and self.device == "cuda":
            logger.info("XTTS2: Acordando modelo. Movendo da CPU para GPU (Memory Offload)...")
            self.model = self.model.to(self.device)
            self._is_offloaded = False
            
        self._reset_idle_timer()

        # OTIMIZAÇÃO DE QUALIDADE (Prosódia)
        # Limpeza leve, mas MANTENDO pontuações vitais.
        text = text.replace('"', '').replace("-", " ").replace(".", "\n")
        text = text.strip()
        # Dica de Ouro XTTS: Sempre force um caractere de encerramento no final 
        # para evitar alucinações ("babbling") da IA.
        if not text.endswith((".", "!", "?")):
            text += "."
            
        try:
            logger.info(f"XTTS2: Synthesizing (in-memory) using speaker {voice_path}")
            
            wav_array = self.model.tts(
                text=text,
                speaker_wav=voice_path,
                language=target_lang,
                temperature=temperature
            )
            
            # Converte o array para bytes WAV na memória RAM (Zero I/O de disco)
            # XTTS v2 tem a taxa de amostragem (sample rate) de 24000 nativa
            buffer = io.BytesIO()
            sf.write(buffer, np.array(wav_array), 24000, format='WAV')
            audio_bytes = buffer.getvalue()
            
            logger.info(f"Engine XTTS2 executed successfully on {self.backend}")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"XTTS2: Execution failed: {e}")
            raise e

    def is_loaded(self):
        return self.model is not None

    def unload(self):
        """Libera os recursos da estratégia."""
        if hasattr(self, '_idle_timer') and self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
            
        if self.model:
            del self.model
            self.model = None
            
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # Limpeza mais agressiva de VRAM
        logger.info("XTTS2: Model unloaded.")