from fastapi import APIRouter, HTTPException
import logging
import re
from pydantic import BaseModel
import os

logger = logging.getLogger("JulyEngine.Routers.Calculator")

router = APIRouter(prefix="/system", tags=["System"])

class ResourceCheckRequest(BaseModel):
    num_params: float # In billions (e.g., 3 for 3B, 0.5 for 500M)
    quantization: str
    context_window: int
    num_layers: int = -1

from ..services.resource_calculator import estimate_vram_ram

@router.post("/check-resources")
async def check_resources(req: ResourceCheckRequest):
    return estimate_vram_ram(req.model_name or "model", req.num_params, req.quantization, req.context_window, req.num_layers)
