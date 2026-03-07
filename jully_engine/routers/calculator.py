from fastapi import APIRouter, HTTPException
import logging
import re
from pydantic import BaseModel

logger = logging.getLogger("JulyEngine.Routers.Calculator")

router = APIRouter(prefix="/system", tags=["System"])

class ResourceCheckRequest(BaseModel):
    num_params: float # In billions (e.g., 3 for 3B, 0.5 for 500M)
    quantization: str
    context_window: int
    num_layers: int = -1

# Base size multipliers for quantizations (roughly bits per weight)
QUANTIZATION_MULTIPLIERS = {
    "Q2_K": 2.5,
    "Q3_K_M": 3.3,
    "Q3_K_L": 3.8,
    "Q4_0": 4.5,
    "Q4_K_M": 4.8,
    "Q4_K_S": 4.5,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.2,
    "Q6_K": 6.5,
    "Q8_0": 8.5,
    "F16": 16.0,
}

def estimate_vram_ram(params_b: float, quant: str, ctx: int, layers: int = -1):
    quant_upper = quant.upper()
    bits_per_weight = None
    
    for key, mult in QUANTIZATION_MULTIPLIERS.items():
        if key in quant_upper:
            bits_per_weight = mult
            break
            
    if bits_per_weight is None:
        # Default fallback if unknown (assume Q4)
        bits_per_weight = 4.5

    # 1 Billion params * bits per weight / 8 = Bytes. Then / 1024^3 = GB.
    # Basically: Params(B) * bits_per_weight / 8 = Size in GB
    model_size_gb = (params_b * bits_per_weight) / 8.0
    
    # Context window estimate: very rough heuristic
    # Usually around 100MB per 1k context for a 7B model
    # We scale it relative to params for an approximation
    ctx_memory_gb = (ctx / 1024) * 0.1 * (params_b / 7.0)
    
    total_required_gb = model_size_gb + ctx_memory_gb
    
    # If layers is -1, assume 100% offload to VRAM
    if layers == -1:
        vram_req = total_required_gb
        ram_req = 0.5 # Minimal base RAM overhead
    elif layers == 0:
        vram_req = 0.0
        ram_req = total_required_gb + 0.5
    else:
        # Simplistic split: assume standard model has ~32 layers. 
        # If user says 16 layers, ~50% goes to VRAM.
        ratio = min(layers / 32.0, 1.0)
        vram_req = total_required_gb * ratio
        ram_req = (total_required_gb * (1 - ratio)) + 0.5

    return {
        "model_size_gb": round(model_size_gb, 2),
        "context_memory_gb": round(ctx_memory_gb, 2),
        "total_required_gb": round(total_required_gb, 2),
        "estimated_vram_gb": round(vram_req, 2),
        "estimated_ram_gb": round(ram_req, 2)
    }

@router.post("/check-resources")
async def check_resources(req: ResourceCheckRequest):
    return estimate_vram_ram(req.num_params, req.quantization, req.context_window, req.num_layers)
