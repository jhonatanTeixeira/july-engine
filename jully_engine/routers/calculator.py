from fastapi import APIRouter, HTTPException
import logging
import re
from pydantic import BaseModel
import os
from typing import Optional

logger = logging.getLogger("JulyEngine.Routers.Calculator")

router = APIRouter(prefix="/system", tags=["System"])

class ResourceCheckRequest(BaseModel):
    model_name: Optional[str] = "model"
    model_id: Optional[str] = None # repo_id
    filename: Optional[str] = None
    num_params: float # In billions (e.g., 3 for 3B, 0.5 for 500M)
    quantization: str
    context_window: int
    num_layers: int = -1
    kv_cache_quantization: Optional[str] = "FP16"
    raw_gguf_meta: Optional[Dict[str, Any]] = None


@router.post("/check-resources")
async def check_resources(req: ResourceCheckRequest):
    from ..services.resource_calculator import estimate_vram_ram
    from ..services.gguf_scanner import GGUFMetadataScanner

    resolved_meta = req.raw_gguf_meta
    
    # Se não mandou o metadado mastigado, tenta resolver agora (Puxa do cache ou remote)
    if not resolved_meta and req.model_id and req.filename:
        resolved_meta = await GGUFMetadataScanner.resolve_metadata(req.model_id, req.filename)

    return estimate_vram_ram(
        req.model_name or "model", 
        req.num_params, 
        req.quantization, 
        req.context_window, 
        layers=req.num_layers,
        metadata=resolved_meta,
        kv_cache_quantization=req.kv_cache_quantization
    )
