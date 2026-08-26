import gc
import psutil
import time
import logging
import os
from typing import Optional

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
        self.gpu_type = None  # 'nvidia' or 'amd'
        self.gpu_name = "Unknown"
        self.total_vram_mb = 0
        self._nvml_handle = None
        self._amd_vram_total_path = None
        self._amd_vram_used_path = None

        # Inicialização Preguiçosa do NVML (nvidia-ml-py)
        try:
            import pynvml
            pynvml.nvmlInit()
            # Index 0 costuma ser a GPU dedicada em notebooks híbridos
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            self.gpu_name = name if isinstance(name, str) else name.decode('utf-8')
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self.total_vram_mb = info.total / 1024**2
            self.has_gpu = True
            self.gpu_type = 'nvidia'
            
            logger.info(f"ResourceManager: NVML Initialized. NVIDIA GPU detected: {self.gpu_name} ({self.total_vram_mb:.2f} MB)")
        except Exception as e:
            logger.debug(f"ResourceManager: NVIDIA GPU not found or NVML error: {e}")
            self._init_amd_gpu()
            
        self.initialized = True

    def _init_amd_gpu(self):
        """Tenta detectar GPU AMD via sysfs (Linux)."""
        try:
            # Busca por cards em /sys/class/drm/
            base_path = "/sys/class/drm"
            if not os.path.exists(base_path):
                return

            for card in sorted(os.listdir(base_path)):
                if not card.startswith("card"):
                    continue
                
                total_path = os.path.join(base_path, card, "device", "mem_info_vram_total")
                used_path = os.path.join(base_path, card, "device", "mem_info_vram_used")
                
                if os.path.exists(total_path):
                    with open(total_path, "r") as f:
                        total_bytes = int(f.read().strip())
                        self.total_vram_mb = total_bytes / 1024**2
                    
                    self._amd_vram_total_path = total_path
                    self._amd_vram_used_path = used_path
                    self.has_gpu = True
                    self.gpu_type = 'amd'
                    self.gpu_name = self._get_amd_gpu_name(card)
                    logger.info(f"ResourceManager: AMD GPU detected via sysfs ({card}): {self.gpu_name} ({self.total_vram_mb:.2f} MB)")
                    break
        except Exception as e:
            logger.warning(f"ResourceManager: Error detecting AMD GPU: {e}")

    def _get_amd_gpu_name(self, card_name: str) -> str:
        """Tenta obter um nome amigável para a GPU AMD."""
        try:
            device_path = f"/sys/class/drm/{card_name}/device/device"
            vendor_path = f"/sys/class/drm/{card_name}/device/vendor"
            
            if os.path.exists(device_path) and os.path.exists(vendor_path):
                with open(device_path, "r") as f:
                    dev_id = f.read().strip()
                with open(vendor_path, "r") as f:
                    ven_id = f.read().strip()
                
                return f"AMD GPU ({ven_id}:{dev_id})"
        except:
            pass
        return "AMD GPU"

    def get_vram_usage(self):
        """Retorna (Free, Used, Total) em MiB, calculando o espaço disponível real."""
        if not self.has_gpu:
            return 0.0, 0.0, 0.0
            
        if self.gpu_type == 'nvidia' and self._nvml_handle:
            try:
                import pynvml
                info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                
                total_mib = info.total / 1024**2
                used_mib = info.used / 1024**2
                free_mib = total_mib - used_mib
                
                return free_mib, used_mib, total_mib
            except Exception as e:
                logger.error(f"ResourceManager: Error reading NVML: {e}")
                
        elif self.gpu_type == 'amd' and self._amd_vram_total_path:
            try:
                with open(self._amd_vram_total_path, "r") as f:
                    total_mib = int(f.read().strip()) / 1024**2
                with open(self._amd_vram_used_path, "r") as f:
                    used_mib = int(f.read().strip()) / 1024**2
                
                free_mib = total_mib - used_mib
                return free_mib, used_mib, total_mib
            except Exception as e:
                logger.error(f"ResourceManager: Error reading AMD sysfs: {e}")

        return 0.0, 0.0, 0.0

    def get_gpu_utilization_percent(self) -> Optional[float]:
        """Retorna a % de utilização de processamento da GPU (0-100), ou None se indisponível."""
        if not self.has_gpu:
            return None

        if self.gpu_type == 'nvidia' and self._nvml_handle:
            try:
                import pynvml
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                return float(rates.gpu)
            except Exception as e:
                logger.error(f"ResourceManager: Error reading NVML utilization: {e}")
                return None

        elif self.gpu_type == 'amd' and self._amd_vram_total_path:
            busy_path = self._amd_vram_total_path.replace("mem_info_vram_total", "gpu_busy_percent")
            try:
                with open(busy_path, "r") as f:
                    return float(f.read().strip())
            except Exception as e:
                logger.warning(f"ResourceManager: Error reading AMD gpu_busy_percent: {e}")
                return None

        return None

    def get_available_vram_mb(self) -> float:
        """Retorna a VRAM livre real. Como sua Intel cuida do Windows, aqui será quase o total."""
        if not self.has_gpu:
            return 0.0
            
        free_mb, used_mb, _ = self.get_vram_usage()
        
        available = free_mb
        
        return max(0.0, available)

    def clear_memory(self):
        """Limpa RAM, VRAM e tenta forçar o fechamento de contextos residuais."""
        import gc
        import ctypes
        
        gc.collect()
        
        try:
            import torch

            # Limpeza genérica de cache se o torch estiver presente
            if hasattr(torch, self.gpu_type if self.gpu_type else 'cuda'):
                backend = getattr(torch, self.gpu_type if self.gpu_type == 'cuda' else 'cuda') # Fallback to cuda check
                # Note: ROCm often uses torch.cuda as well. 
                # If using something else, we might need more specific checks.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                if hasattr(obj, 'is_cuda') and obj.is_cuda:
                                    del obj
                        except:
                            pass
                    
                    torch.cuda.synchronize()
                    logger.info(f"ResourceManager: {self.gpu_type.upper() if self.gpu_type else 'GPU'} VRAM cleanup completed")
                elif hasattr(torch, 'xpu') and torch.xpu.is_available(): # Intel/Generic
                    torch.xpu.empty_cache()
            
            # Se for AMD e não estiver usando torch.cuda (ROCm), talvez use outro backend no futuro
            # Por enquanto, empty_cache() no cuda é o padrão para ROCm.
            
        except Exception as e:
            logger.error(f"Error during VRAM cleanup: {e}")

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