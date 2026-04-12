import os
import re
import logging

logger = logging.getLogger("JulyEngine.Services.ResourceCalculator")

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

def estimate_vram_ram(
    combined_name: str, 
    params_b: float, 
    quant: str, 
    ctx: int, 
    layers: int = -1, 
    total_layers: int = -1,
    metadata: dict = None,
    kv_cache_quantization: str = "FP16"
):
    """
    Estimates VRAM and RAM requirements for an LLM model running via GGUF.
    Logic shared between the router (API) and engine models (orchestrator).
    """
    quant_upper = str(quant).upper()
    bits_per_weight = None
    
    for key, mult in QUANTIZATION_MULTIPLIERS.items():
        if key in quant_upper:
            bits_per_weight = mult
            break
            
    if bits_per_weight is None:
        bits_per_weight = 4.5 # Default fallback (assume Q4)

    # 1. PESO ESTÁTICO DO MODELO (Todos os parâmetros contam, mesmo em MoE)
    # Params(B) * bits_per_weight / 8 = Size in GB
    model_size_gb = (params_b * bits_per_weight) / 8.0
    
    # 2. CÁLCULO DO KV CACHE (Determinístico via GQA se houver metadata)
    # BytesPerElement: FP16=2, Q8_0=1, Q4_0=0.5
    bytes_per_element = 2.0
    kv_quant_upper = str(kv_cache_quantization).upper()
    if "Q8_0" in kv_quant_upper:
        bytes_per_element = 1.0
    elif "Q4_0" in kv_quant_upper:
        bytes_per_element = 0.5
        
    if metadata and all(metadata.get(k, 0) > 0 for k in ["block_count", "head_count_kv", "embedding_length", "head_count"]):
        # FÓRMULA GQA: 2 * Layers * Heads_kv * (Embedding / Heads_total) * BytesPerElement
        layers_count = metadata["block_count"]
        heads_kv = metadata["head_count_kv"]
        embedding = metadata["embedding_length"]
        heads_total = metadata["head_count"]
        
        # Memory per token in bytes
        memory_per_token = 2 * layers_count * heads_kv * (embedding / heads_total) * bytes_per_element
        ctx_memory_gb = (memory_per_token * ctx) / (1024**3)
        
        # Update total_layers if we have real data
        if total_layers == -1:
            total_layers = layers_count
            
    else:
        # FALLBACK: Estimativa base do KV Cache baseada em parâmetros ativos (mais imprecisa)
        combined_name_lower = combined_name.lower()
        active_params = params_b 
        
        if "mixtral" in combined_name_lower or "moe" in combined_name_lower:
            active_params = params_b / 3.5
            
        match_active = re.search(r'-a(\d+(?:\.\d+)?)b', combined_name_lower)
        if match_active:
            active_params = float(match_active.group(1))

        # Estimativa legada baseada em params/7.0
        base_ctx_memory_gb = (ctx / 1024) * 0.1 * (active_params / 7.0)
        
        # Ajuste de bits (FP16 assume 2 bytes por elemento)
        ctx_memory_gb = base_ctx_memory_gb * (bytes_per_element / 2.0)
        
    total_required_gb = model_size_gb + ctx_memory_gb
    
    # 3. DIVISÃO VRAM vs RAM (Baseado em Layers reais)
    if total_layers == -1: total_layers = 32 # Fallback genérico se tudo falhar
    
    if layers == -1:
        vram_req = total_required_gb
        ram_req = 0.5 # Minimal base RAM overhead
    elif layers == 0:
        vram_req = 0.0
        ram_req = total_required_gb + 0.5
    else:
        # Usando a quantidade REAL de layers do modelo para achar o ratio matemático
        safe_total_layers = max(total_layers, 1) 
        ratio = min(layers / float(safe_total_layers), 1.0)
        
        vram_req = total_required_gb * ratio
        ram_req = (total_required_gb * (1 - ratio)) + 0.5

    return {
        "model_size_gb": round(model_size_gb, 2),
        "context_memory_gb": round(ctx_memory_gb, 2),
        "total_required_gb": round(total_required_gb, 2),
        "estimated_vram_gb": round(vram_req, 2),
        "estimated_ram_gb": round(ram_req, 2),
        "total_layers": total_layers
    }