import os
import io
import logging
from typing import Any, Dict, Optional
import numpy as np
import soundfile as sf
import threading

logger = logging.getLogger("JulyEngine.Models.FasterQwen3TTS")

class FasterQwen3TTS:
    """
    Faster Qwen3-TTS model para síntese de alta velocidade.
    Otimizado para máxima economia de VRAM em GPUs consumer-grade.
    """
    def __init__(self, backend="gpu", model_size="0.6B"):
        self.backend = backend
        self.device = "cpu" # Default until load()
        self.model = None
        self.model_size = model_size # "0.6B" (Recomendado para restrição de VRAM) ou "1.7B"
        
        # OFF-LOAD AND IDLE CONFIGS
        # Tempo reduzido para 60s: Libera a VRAM mais rápido para outros agentes/modelos da engine
        self.idle_timeout = 60  
        self._idle_timer = None
        self._is_offloaded = False

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        """Calcula a VRAM estimada. O 0.6B em bf16 consome ~2.6GB + overhead."""
        if self.backend == "cpu":
            return 0

        return 3000 if self.model_size == "0.6B" else 6500

    def _reset_idle_timer(self):
        if self.device == "cpu":
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self.idle_timeout, self._perform_idle_offload)
        self._idle_timer.start()

    def _perform_idle_offload(self):
        if self.model is not None and not self._is_offloaded:
            logger.info("FasterQwen3TTS: Timeout ocioso atingido. Aplicando Idle Offload (GPU -> CPU)...")
            self.model = self.model.to("cpu")
            self._is_offloaded = True
            
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load(self):
        if self.model is None:
            import torch
            from faster_qwen3_tts import Qwen3TTS # Importação da lib
            
            self.device = "cuda" if self.backend == "gpu" and torch.cuda.is_available() else "cpu"
            
            try:
                logger.info(f"FasterQwen3TTS: Loading {self.model_size} model on {self.device}")
                
                # OTIMIZAÇÃO VRAM 1: Carregar nativamente no formato mais econômico suportado
                # A arquitetura Ampere suporta bfloat16 perfeitamente, evitando overflows do fp16
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                
                self.model = Qwen3TTS.from_pretrained(
                    f"Qwen/Qwen3-TTS-{self.model_size}",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True # Otimiza o carregamento inicial RAM -> VRAM
                )
                
                self.model = self.model.to(self.device)
                
                # OTIMIZAÇÃO VRAM 2: Forçar o modelo para modo de avaliação
                # Garante que nenhum buffer de dropout ou batchnorm aloque memória extra
                self.model.eval() 
                
                self._is_offloaded = False
                logger.info("FasterQwen3TTS loaded successfully.")
            except Exception as e:
                logger.error(f"FasterQwen3TTS: Failed to load: {e}")
                raise e

    def run(self, text: str, voice_id: str, language: str, temperature: float = 0.7) -> bytes:
        from ..services.voice_service import voice_service
        
        voice_res = voice_service.get_voice_path(voice_id)
        
        if not voice_res:
            logger.error(f"FasterQwen3TTS: Voice {voice_id} not found and no fallback available.")
            raise ValueError(f"Voice {voice_id} not found")
            
        voice_path, voice_lang = voice_res
        target_lang = language or voice_lang

        if self.model is None:
            self.load()
        elif self._is_offloaded and self.device == "cuda":
            logger.info("FasterQwen3TTS: Acordando modelo. Movendo da CPU para GPU...")
            self.model = self.model.to(self.device)
            self._is_offloaded = False
            
        self._reset_idle_timer()

        # Limpeza de texto focada na estabilidade da arquitetura do Qwen3
        text = text.replace('"', '').replace("-", " ").strip()
        if not text.endswith((".", "!", "?")):
            text += "."
            
        try:
            logger.info(f"FasterQwen3TTS: Synthesizing using speaker {voice_path}")
            import torch
            
            # OTIMIZAÇÃO VRAM 3: inference_mode() desliga completamente o tracking de autograd.
            # É mais restrito que no_grad() e economiza blocos inteiros de memória do PyTorch.
            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    # A API do faster-qwen devolve o array e o sample rate real do modelo
                    wav_array, sample_rate = self.model.synthesize(
                        text=text,
                        speaker_ref=voice_path,
                        lang=target_lang,
                        temperature=temperature
                    )
            
            # Conversão in-memory
            buffer = io.BytesIO()
            sf.write(buffer, np.array(wav_array), sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            # OTIMIZAÇÃO VRAM 4: Limpeza síncrona pós-inferência
            # Essencial para evitar que tensores isolados do KV cache vazem na VRAM após a geração
            del wav_array
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            logger.info(f"Engine FasterQwen3TTS executed successfully on {self.backend}")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"FasterQwen3TTS: Execution failed: {e}")
            raise e

    def is_loaded(self):
        return self.model is not None

    def unload(self):
        """Libera os recursos e zera a VRAM retida."""
        if hasattr(self, '_idle_timer') and self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
            
        if self.model:
            del self.model
            self.model = None
        from ..resource_manager import resource_manager
        resource_manager.clear_memory()

        logger.info("FasterQwen3TTS: Model unloaded.")