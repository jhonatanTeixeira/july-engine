import os
import psutil
import logging
import json
from fastapi import APIRouter
from typing import Dict, Any, List, Optional
from ..resource_manager import resource_manager

logger = logging.getLogger("JulyEngine.Routers.Monitoring")

router = APIRouter(prefix="/system/monitoring", tags=["Monitoring"])

def get_gpu_info():
    if os.environ.get("DISABLE_GPU", "false").lower() == "true":
        return None
    
    if not resource_manager.has_gpu:
        return []
    
    try:
        free_mib, used_mib, total_mib = resource_manager.get_vram_usage()
        
        # We maintain the list format for backward compatibility
        devices = [{
            "index": 0,
            "name": resource_manager.gpu_name,
            "vram_total_mb": total_mib,
            "vram_used_mb": used_mib,
            "vram_free_mb": free_mib,
            "graphics_clock_mhz": None, # Clocks are not currently tracked by ResourceManager
            "memory_clock_mhz": None
        }]
        
        return devices
    except Exception as e:
        logger.warning(f"Failed to get GPU info from ResourceManager: {e}")
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


@router.get("/models/{alias}")
async def model_health(alias: str):
    """Status real de um preset GGUF pelo alias -- seqs ativos/total, %
    GPU, VRAM livre (AgentGateway não faz polling ativo aqui; o failover
    real é via 503 do próprio /v1/openai/chat/completions com o header
    x-health-max-seq, ver llama_gguf.py::GGUF.run_chat). Serve pra
    dashboard/checagem manual."""
    from ..model_loader import model_loader
    from ..services.models_service import model_service

    health = {"loaded": False, "active_seqs": 0, "total_seqs": 0, "saturated": False}
    for instance in model_loader.instances.values():
        meta = getattr(instance, "meta", None) or {}
        if meta.get("model_alias") == alias or meta.get("alias") == alias:
            if hasattr(instance, "get_health_status"):
                health = instance.get_health_status()
            break
    else:
        model_row = model_service.backend.get_model(alias) or {}
        if model_row:
            health["total_seqs"] = model_row.get("n_seq_max", 0)

    return {
        "alias": alias,
        **health,
        "gpu_utilization_percent": resource_manager.get_gpu_utilization_percent(),
        "vram_free_mb": resource_manager.get_available_vram_mb(),
    }
