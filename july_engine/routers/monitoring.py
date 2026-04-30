import os
import psutil
import logging
import json
from fastapi import APIRouter
from typing import Dict, Any, List, Optional

logger = logging.getLogger("JulyEngine.Routers.Monitoring")

router = APIRouter(prefix="/system/monitoring", tags=["Monitoring"])

def get_gpu_info():
    if os.environ.get("DISABLE_GPU", "false").lower() == "true":
        return None
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Try to get clocks
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                graphics_clock = None
                mem_clock = None

            devices.append({
                "index": i,
                "name": name if isinstance(name, str) else name.decode('utf-8'),
                "vram_total_mb": memory.total / 1024 / 1024,
                "vram_used_mb": memory.used / 1024 / 1024,
                "vram_free_mb": memory.free / 1024 / 1024,
                "graphics_clock_mhz": graphics_clock,
                "memory_clock_mhz": mem_clock
            })
        pynvml.nvmlShutdown()
        return devices
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return []

def get_cpu_info():
    if os.environ.get("DISABLE_CPU", "false").lower() == "true":
        return None
    
    cpu_data = {
        "usage_percent": psutil.cpu_percent(interval=0.1),
        "count_logical": psutil.cpu_count(logical=True),
        "count_physical": psutil.cpu_count(logical=False),
        "metadata": {}
    }
    
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        cpu_data["metadata"] = {
            "brand": info.get("brand_raw"),
            "arch": info.get("arch"),
            "bits": info.get("bits"),
            "features": info.get("flags", [])[:10], # Top 10 flags
            "hz_advertised": info.get("hz_advertised_friendly"),
            "hz_actual": info.get("hz_actual_friendly")
        }
    except Exception as e:
        logger.warning(f"Failed to get detailed CPU info: {e}")
        
    return cpu_data

def get_ram_info():
    ram = psutil.virtual_memory()
    return {
        "total_gb": ram.total / 1024 / 1024 / 1024,
        "available_gb": ram.available / 1024 / 1024 / 1024,
        "used_gb": ram.used / 1024 / 1024 / 1024,
        "usage_percent": ram.percent
    }

@router.get("/")
async def system_monitoring():
    return {
        "gpu": get_gpu_info(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info()
    }
