import gc
import psutil
import time
import logging
import os

logger = logging.getLogger("JulyEngine.ResourceManager")


class ResourceManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.has_gpu = False
        self.total_vram_mb = 0
        self._nvml_handle = None

        # Inicialização Preguiçosa do NVML (nvidia-ml-py)
        try:
            import pynvml
            pynvml.nvmlInit()
            # Index 0 costuma ser a GPU dedicada em notebooks híbridos
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self.total_vram_mb = info.total / 1024**2
            self.has_gpu = True
            
            logger.info(f"ResourceManager: NVML Initialized. NVIDIA GPU detected: {self.total_vram_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"ResourceManager: NVML not available or No NVIDIA GPU found: {e}")
            
        self.initialized = True

    def get_vram_usage(self):
        """Retorna (Free, Used, Total) em Megabytes direto do Driver."""
        if not self.has_gpu or not self._nvml_handle:
            return 0.0, 0.0, 0.0
            
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return (
                info.free / 1024**2,
                info.used / 1024**2,
                info.total / 1024**2
            )
        except Exception as e:
            logger.error(f"ResourceManager: Error reading NVML: {e}")
            return 0.0, 0.0, 0.0

    def get_available_vram_mb(self) -> float:
        """Retorna a VRAM livre real. Como sua Intel cuida do Windows, aqui será quase o total."""
        if not self.has_gpu:
            return 0.0
            
        free_mb, used_mb, _ = self.get_vram_usage()
        
        # Margem de segurança mínima (100MB) para o contexto do driver CUDA não travar
        safety_margin = 100 
        available = free_mb - safety_margin
        
        return max(0.0, available)

    def clear_memory(self):
        """Limpa RAM e solicita ao PyTorch que libere cache da VRAM."""
        gc.collect()
        
        # Import local para não pesar o topo do arquivo
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("ResourceManager: PyTorch CUDA cache cleared")
        except ImportError:
            pass
            
        time.sleep(0.1) # Breve pausa para o driver atualizar as métricas

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_ram_usage(self) -> float:
        return psutil.virtual_memory().percent

    def get_available_ram_mb(self) -> float:
        return psutil.virtual_memory().available / 1024**2

    def check_ram_headroom(self, required_mb: float) -> bool:
        return self.get_available_ram_mb() > required_mb
        
    def check_memory_headroom(self, required_mb: float) -> bool:
        """Verifica se a VRAM necessária cabe na GPU dedicada."""
        available = self.get_available_vram_mb()
        logger.info(f"ResourceManager: Check VRAM - Available: {available:.2f} MB, Required: {required_mb:.2f} MB")
        return available >= required_mb


# Instância única para todo o sistema
resource_manager = ResourceManager()