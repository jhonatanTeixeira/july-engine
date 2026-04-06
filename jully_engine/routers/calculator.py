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
    num_params: float # In billions (e.g., 3 for 3B, 0.5 for 500M)
    quantization: str
    context_window: int
    num_layers: int = -1


@router.post("/check-resources")
async def check_resources(req: ResourceCheckRequest):
    from ..services.resource_calculator import estimate_vram_ram

    return estimate_vram_ram(req.model_name or "model", req.num_params, req.quantization, req.context_window, req.num_layers)
