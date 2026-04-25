from fastapi import APIRouter, HTTPException
import logging
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ..services.resource_calculator import estimate_vram_ram

logger = logging.getLogger("JulyEngine.Routers.Calculator")
router = APIRouter(prefix="/system", tags=["System"])

class ResourceCheckRequest(BaseModel):
    model_path: Optional[str] = "model"
    model_id: Optional[str] = None # repo_id
    filename: Optional[str] = None
    context_window: str | int = "4k"
    gpu_layers: Optional[int] = -1
    kv_cache_quantization: Optional[str] = "FP16"
    mmproj_path: Optional[str] = None
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    flash_attn: Optional[bool] = True

@router.post("/check-resources")
async def check_resources(req: ResourceCheckRequest):
    """
    Unified entry point for VRAM/RAM estimation. 
    Supports local paths, HF cache, or remote scan.
    """
    return await estimate_vram_ram(
        model_path=req.model_path,
        context_window=req.context_window,
        kv_cache_quantization=req.kv_cache_quantization,
        gpu_layers=req.gpu_layers if req.gpu_layers != -1 else None,
        repo_id=req.model_id,
        filename=req.filename,
        mmproj_path=req.mmproj_path,
        mmproj_repo_id=req.mmproj_id,
        mmproj_filename=req.mmproj_filename
    )
