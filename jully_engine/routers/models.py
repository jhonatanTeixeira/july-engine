import os
import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

logger = logging.getLogger("JulyEngine.Routers.Models")

router = APIRouter(prefix="/models/gguf", tags=["Models"])

# Define cache dir and models.json path
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
MODELS_JSON_PATH = os.path.join(CACHE_DIR, "july_models.json")

# ==========================================
# 🧠 MATRIZ DE CONHECIMENTO (HEURÍSTICAS)
# ==========================================
MODEL_HEURISTICS = [
    {
        "keywords": ["deepseek-r1", "r1-distill"],
        "metadata": {"force_reasoning": True}
    },
    {
        "keywords": ["llama-3", "llama3"],
        "metadata": {"template": "llama-3"}
    },
    {
        "keywords": ["qwen"],
        "metadata": {"template": "qwen"}
    },
    {
        "keywords": ["mistral", "mixtral"],
        "metadata": {"template": "mistral"}
    },
    {
        "keywords": ["chatml", "hermes"],
        "metadata": {"template": "chatml"}
    },
    {
        "keywords": ["llava", "nanollava", "moondream", "pixtral"],
        "metadata": {"model_type": "vision"}
    }
]

# ==========================================
# FUNÇÕES DE DETECÇÃO INTELIGENTE
# ==========================================
def guess_params_and_quant(filename: str):
    params = None
    quant = None
    
    # Busca por B (bilhões) ex: 7b, 7B, 0.5b
    param_match = re.search(r'([\d\.]+)[Bb]', filename)
    if param_match:
        params = float(param_match.group(1))

    # Busca por quantização padrão GGUF ex: q4_k_m, Q8_0, f16
    quant_match = re.search(r'([QqFf]\d+_[Kk_0-9MmSsL]+|[Qq]\d|fp16|fp32)', filename, re.IGNORECASE)
    if quant_match:
        quant = quant_match.group(1).upper()

    return params, quant

def detect_model_metadata(model_id: str, filename: str) -> Dict[str, Any]:
    """Cruza o nome do repositório e do arquivo com a matriz de heurísticas."""
    combined_name = f"{model_id} {filename}".lower()
    
    # Defaults
    detected = {
        "model_type": "text",
        "template": "chatml", # Fallback seguro geral
        "force_reasoning": False,
        "num_params": None,
        "quantization": None
    }
    
    # Params e Quantização (Regex puro)
    p, q = guess_params_and_quant(filename)
    detected["num_params"] = p
    detected["quantization"] = q
    
    # Varre a Matriz de Conhecimento
    for rule in MODEL_HEURISTICS:
        if any(keyword in combined_name for keyword in rule["keywords"]):
            detected.update(rule["metadata"])
            
    # Casos Especiais Combinados (ex: DeepSeek-R1-Distill-Llama-8B)
    if "deepseek-r1" in combined_name:
        detected["force_reasoning"] = True
        if "llama" in combined_name:
            detected["template"] = "llama-3"
        elif "qwen" in combined_name:
            detected["template"] = "qwen"

    return detected

# ==========================================
# MODELOS DE DADOS (PYDANTIC)
# ==========================================
class DetectRequest(BaseModel):
    model_id: str
    filename: str

class DownloadRequest(BaseModel):
    model_alias: str
    model_type: str # "text" or "vision"
    model_id: str # Hugging Face repo id
    hf_id: Optional[str] = None # Alias for model_id
    filename: str
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    template: Optional[str] = None
    context_window: Optional[int] = 2048
    num_params: Optional[float] = None # In billions
    quantization: Optional[str] = None
    num_layers: Optional[int] = -1
    force_reasoning: Optional[bool] = None

class UpdateMetadataRequest(BaseModel):
    model_type: Optional[str] = None
    model_id: Optional[str] = None
    filename: Optional[str] = None
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    template: Optional[str] = None
    context_window: Optional[int] = None
    num_params: Optional[float] = None
    quantization: Optional[str] = None
    num_layers: Optional[int] = None
    force_reasoning: Optional[bool] = None

# ==========================================
# BANCO DE DADOS (JSON)
# ==========================================
def load_models_db() -> Dict[str, Any]:
    if os.path.exists(MODELS_JSON_PATH):
        try:
            with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {MODELS_JSON_PATH}: {e}")
            data = {}
        
        data.setdefault('xtts', {
            'model_type': 'tts',
            'estimated_vram': 3000
        })
        
        data.setdefault('faster-whisper', {
            'model_type': 'stt',
            'estimated_vram': 1500            
        })
        
        return data
    
    return {}

def save_models_db(db: Dict[str, Any]):
    os.makedirs(os.path.dirname(MODELS_JSON_PATH), exist_ok=True)
    with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)

# ==========================================
# ENDPOINTS
# ==========================================

@router.post("/detect_metadata")
async def api_detect_metadata(request: DetectRequest):
    """Detecta heurísticas do modelo baseado no nome e arquivo."""
    metadata = detect_model_metadata(request.model_id, request.filename)
    return {
        "status": "success",
        "detected_metadata": metadata
    }

@router.post("/download")
async def download_gguf(request: DownloadRequest):
    # Pre-validation for vision models
    if request.model_type == "vision":
        if not request.mmproj_id or not request.mmproj_filename:
            async def error_gen():
                yield f"data: {json.dumps({'status': 'error', 'message': 'mmproj_id e mmproj_filename são obrigatórios para modelos de visão'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def progress_generator():
        try:
            yield f"data: {json.dumps({'status': 'starting', 'message': f'Iniciando download de {request.model_alias}'})}\n\n"
            
            # Helper para preencher dinamicamente o que o usuário deixou em branco
            auto_meta = detect_model_metadata(request.model_id, request.filename)
            final_params = request.num_params if request.num_params is not None else auto_meta["num_params"]
            final_quant = request.quantization if request.quantization is not None else auto_meta["quantization"]
            final_template = request.template if request.template else auto_meta["template"]
            final_reasoning = request.force_reasoning if request.force_reasoning is not None else auto_meta["force_reasoning"]

            yield f"data: {json.dumps({'status': 'downloading', 'message': 'Baixando arquivo principal...'})}\n\n"
            
            # Download the main file (blocking call wrapped in thread to not block event loop)
            loop = asyncio.get_event_loop()
            file_path = await loop.run_in_executor(None, lambda: hf_hub_download(
                repo_id=request.model_id, 
                filename=request.filename,
                cache_dir=CACHE_DIR
            ))
            
            mmproj_path = None
            if request.model_type == "vision":
                yield f"data: {json.dumps({'status': 'downloading', 'message': 'Baixando componente de visão (mmproj)...'})}\n\n"
                mmproj_path = await loop.run_in_executor(None, lambda: hf_hub_download(
                    repo_id=request.mmproj_id, 
                    filename=request.mmproj_filename,
                    cache_dir=CACHE_DIR
                ))

            # Update DB
            db = load_models_db()
            db[request.model_alias] = {
                "model_alias": request.model_alias,
                "model_type": request.model_type,
                "model_id": request.model_id,
                "filename": request.filename,
                "mmproj_id": request.mmproj_id,
                "mmproj_filename": request.mmproj_filename,
                "template": final_template,
                "context_window": request.context_window,
                "num_params": final_params,
                "quantization": final_quant,
                "num_layers": request.num_layers,
                "force_reasoning": final_reasoning,
                "file_path": file_path,
                "mmproj_path": mmproj_path
            }
            save_models_db(db)

            yield f"data: {json.dumps({'status': 'success', 'message': 'Download concluído!', 'metadata': db[request.model_alias]})}\n\n"
        except Exception as e:
            logger.error(f"Download failed: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(progress_generator(), media_type="text/event-stream")

@router.get("/")
async def list_gguf_models():
    db = load_models_db()
    return {"models": list(db.values())}

@router.put("/{model_alias}")
async def update_model_metadata(model_alias: str, request: UpdateMetadataRequest):
    db = load_models_db()
    if model_alias not in db:
        raise HTTPException(status_code=404, detail=f"Model {model_alias} not found")
        
    model_data = db[model_alias]
    
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        model_data[key] = value
        
    db[model_alias] = model_data
    save_models_db(db)
    
    return {"status": "success", "model": model_data}

@router.delete("/{model_alias}")
async def delete_model(model_alias: str):
    db = load_models_db()
    if model_alias in db:
        del db[model_alias]
        save_models_db(db)
        return {"status": "success", "message": f"Model {model_alias} removed from database"}
    raise HTTPException(status_code=404, detail="Model not found")