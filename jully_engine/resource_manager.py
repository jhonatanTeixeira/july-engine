import gc
import psutil
import time
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
        
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.has_gpu = self.device == "cuda"
        self.total_vram = 0
        
        # Inicialização PyTorch Nativa (Fim do NVML/pynvml)
        if self.has_gpu:
            try:
                # mem_get_info retorna (free_bytes, total_bytes) direto do driver nativo
                free, total = torch.cuda.mem_get_info()
                self.total_vram = total
                logger.info(f"ResourceManager: PyTorch Native Memory Management active. Total VRAM: {self.total_vram / 1024**2:.2f} MB")
            except Exception as e:
                logger.warning(f"ResourceManager: PyTorch CUDA ready, but mem_get_info failed: {e}")
        else:
            logger.warning("ResourceManager: No compatible NVIDIA GPU or PyTorch CUDA found.")
            
        self.initialized = True

    def get_vram_usage(self):
        """Retorna as métricas em Megabytes: (OS_Free, Torch_Allocated, Torch_Reserved)"""
        if not self.has_gpu:
            return 0.0, 0.0, 0.0
            
        try:
            # Visão do Sistema Operacional (Substitui o NVML)
            free_bytes, _ = torch.cuda.mem_get_info()
            system_free = free_bytes / 1024**2

            # Visão do Caching Allocator do PyTorch
            torch_allocated = torch.cuda.memory_allocated() / 1024**2
            torch_reserved = torch.cuda.memory_reserved() / 1024**2

            return system_free, torch_allocated, torch_reserved
        except Exception as e:
            logger.error(f"ResourceManager: Error in get_vram_usage: {e}")
            return 0.0, 0.0, 0.0

    def get_available_vram_mb(self) -> float:
        if not self.has_gpu:
            return 0.0
            
        system_free, allocated, reserved = self.get_vram_usage()
        safety = 200
        
        # A Mágica do PyTorch: A memória "real free" é o que o SO tem livre,
        # MAIS o espaço que o PyTorch já reservou mas que está vazio por dentro.
        real_free_mb = system_free + (reserved - allocated) - safety
        return real_free_mb

    def clear_memory(self):
        """Limpa agressivamente a memória RAM e VRAM."""
        gc.collect()
        
        if self.has_gpu:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("ResourceManager: CUDA cache cleared natively")
            time.sleep(0.2) # Pausa tática para o driver do Windows atualizar

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_ram_usage(self) -> float:
        return psutil.virtual_memory().percent

    def get_available_ram_mb(self) -> float:
        mem = psutil.virtual_memory()
        return mem.available / 1024**2

    def check_ram_headroom(self, required_mb: int) -> bool:
        return self.get_available_ram_mb() > required_mb
        
    def check_memory_headroom(self, required_mb: float) -> bool:
        available = self.get_available_vram_mb()
        logger.info(f"Available VRAM: {available:.2f} MB, Required: {required_mb} MB")
        return available > required_mb

resource_manager = ResourceManager()