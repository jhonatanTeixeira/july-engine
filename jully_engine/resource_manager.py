import gc
import psutil
import os
import time
import logging

try:
    import torch
    import pynvml
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResourceManager")

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
        
        if self.has_gpu:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.total_vram = info.total
                logger.info(f"Detected GPU with {self.total_vram / 1024**3:.2f} GB VRAM")
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                self.has_gpu = False
        
        self.initialized = True

    def get_vram_usage(self):
        if not self.has_gpu:
            return 0, 0, 0, 0
        try:
            # NVML gives system-wide view
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            system_used = info.used
            system_free = info.free

            # Torch gives process-specific view
            torch_allocated = torch.cuda.memory_allocated()
            torch_reserved = torch.cuda.memory_reserved()

            res = (system_used, system_free, torch_allocated, torch_reserved)
            return res
        except Exception as e:
            logger.error(f"Error in get_vram_usage: {e}")
            return 0, 0, 0, 0

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_ram_usage(self) -> float:
        return psutil.virtual_memory().percent

    def get_ram_info(self):
        mem = psutil.virtual_memory()
        return mem.used, mem.available

    def get_vram_info(self):
        if not self.has_gpu:
            return None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "total": info.total / 1024**2,
                "free": info.free / 1024**2,
                "used": info.used / 1024**2
            }
        except Exception:
            return None

    def get_available_ram_mb(self) -> float:
        mem = psutil.virtual_memory()
        return mem.available / 1024**2

    def check_ram_headroom(self, required_mb: int) -> bool:
        return self.get_available_ram_mb() > required_mb

    def clear_memory(self):
        """Aggressively clears memory."""
        gc.collect()
        if self.has_gpu and HAS_TORCH:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
    def check_memory_headroom(self, required_mb: int) -> bool:
        """Checks if there is enough VRAM available."""
        if not self.has_gpu:
            return False
        _, free, _, _ = self.get_vram_usage()
        free_mb = free / 1024**2
        logger.info(f"Available VRAM: {free_mb:.2f} MB, Required: {required_mb} MB")
        return free_mb > required_mb

    def get_available_vram_mb(self) -> float:
        if not self.has_gpu:
            return 0.0
        # system_used, system_free, torch_allocated, torch_reserved
        _, free, allocated, reserved = self.get_vram_usage()
        
        # Real free is system_free + (reserved - allocated)
        # Because (reserved - allocated) can be freed back to OS via empty_cache()
        real_free_mb = (free + (reserved - allocated)) / 1024**2
        return real_free_mb

resource_manager = ResourceManager()
