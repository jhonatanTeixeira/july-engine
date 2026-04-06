import os
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

def estimate_vram_ram(combined_name: str, params_b: float, quant: str, ctx: int, layers: int = -1, total_layers: int = 32):
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
    
    # 2. CÁLCULO DO KV CACHE (A Mágica do Contexto)
    # Se for MoE, o KV Cache escala apenas com os parâmetros ATIVOS na Atenção,
    # que costumam ser grosseiramente ~1/3.5 do total em arquiteturas como Mixtral.
    is_moe = "mixtral" in combined_name.lower() or "moe" in combined_name.lower()
    effective_kv_params = params_b / 3.5 if is_moe else params_b
    
    # Estimativa base do KV Cache em FP16 (16-bits): ~100MB por 1k ctx para um modelo 7B
    base_ctx_memory_gb = (ctx / 1024) * 0.1 * (effective_kv_params / 7.0)
    
    # Lendo a variável de ambiente para compactação de cache
    kv_quant = str(os.environ.get('KV_CACHE_QUANTIZATION', '16'))
    
    # Aplicando os multiplicadores de economia de VRAM do KV Cache
    if kv_quant == '8':
        ctx_memory_gb = base_ctx_memory_gb * 0.5  # Q8_0 pesa metade do FP16
    elif kv_quant == '4':
        ctx_memory_gb = base_ctx_memory_gb * 0.25 # Q4_0 pesa um quarto do FP16
    else:
        ctx_memory_gb = base_ctx_memory_gb        # FP16 (Tamanho normal)
        
    total_required_gb = model_size_gb + ctx_memory_gb
    
    # 3. DIVISÃO VRAM vs RAM (Baseado em Layers reais)
    if layers == -1:
        vram_req = total_required_gb
        ram_req = 0.5 # Minimal base RAM overhead
    elif layers == 0:
        vram_req = 0.0
        ram_req = total_required_gb + 0.5
    else:
        # Usando a quantidade REAL de layers do modelo para achar o ratio matemático
        safe_total_layers = max(total_layers, 1) # Previne divisão por zero
        ratio = min(layers / float(safe_total_layers), 1.0)
        
        vram_req = total_required_gb * ratio
        ram_req = (total_required_gb * (1 - ratio)) + 0.5

    return {
        "model_size_gb": round(model_size_gb, 2),
        "context_memory_gb": round(ctx_memory_gb, 2),
        "total_required_gb": round(total_required_gb, 2),
        "estimated_vram_gb": round(vram_req, 2),
        "estimated_ram_gb": round(ram_req, 2)
    }
