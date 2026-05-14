import os
import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, list_repo_files

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
def detect_model_metadata(model_id: str, filename: str) -> Dict[str, Any]:
    """Cruza o nome do repositório e do arquivo com a matriz de heurísticas."""
    combined_name = f"{model_id} {filename}".lower()
    
    # Defaults
    detected = {
        "model_type": "text",
        "template": "chatml", # Fallback seguro geral
        "force_reasoning": False,
        "n_seq_max": 1,
        "offload_kqv": True,
        "kv_unified": True,
        "logits_all": False,
    }
    
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
    context_window: Optional[int] = 4096
    kv_cache_quantization: Optional[str] = "FP16"
    num_layers: Optional[int] = -1
    force_reasoning: Optional[bool] = None
    is_vision: Optional[bool] = None
    flash_attn: Optional[bool] = True
    n_seq_max: Optional[int] = 1
    offload_kqv: Optional[bool] = True
    kv_unified: Optional[bool] = True
    logits_all: Optional[bool] = False
    vision_on_cpu: Optional[bool] = False

class UpdateMetadataRequest(BaseModel):
    model_type: Optional[str] = None
    model_id: Optional[str] = None
    filename: Optional[str] = None
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    template: Optional[str] = None
    context_window: Optional[int] = None
    kv_cache_quantization: Optional[str] = None
    num_layers: Optional[int] = None
    force_reasoning: Optional[bool] = None
    is_vision: Optional[bool] = None
    flash_attn: Optional[bool] = None
    n_seq_max: Optional[int] = None
    offload_kqv: Optional[bool] = None
    kv_unified: Optional[bool] = None
    logits_all: Optional[bool] = None
    vision_on_cpu: Optional[bool] = None
    file_path: Optional[str] = None
    mmproj_path: Optional[str] = None

class WarmupItem(BaseModel):
    task_type: str  # text_chat, vision_chat, tts, stt, embeddings, image_generation, web_search, etc.
    model: str  # model alias/tag
    backend: Optional[str] = None  # gpu, cpu, api — se omitido, resolve via config

class WarmupRequest(BaseModel):
    models: List[WarmupItem]

# ==========================================
# BANCO DE DADOS (PERSISTENCE)
# ==========================================
from ..services.models_service import ModelsService
models_service = ModelsService()

def load_models_db() -> Dict[str, Any]:
    models = models_service.get_all()
    # convert list to dict
    db = {m.get("model_alias"): m for m in models if m.get("model_alias")}
    
    return db

def save_models_db(db: Dict[str, Any]):
    for alias, data in db.items():
        if alias in ['xtts', 'faster-whisper']:
            continue
        models_service.set(alias, data)


# ==========================================
# ENDPOINTS
# ==========================================

@router.post("/detect_metadata")
async def api_detect_metadata(request: DetectRequest):
    """Detecta heurísticas e metadados reais do GGUF via Smart Resolver."""
    from ..services.resource_calculator import ModelMetadata
    
    # 1. Heurísticas baseadas em nome (Regex)
    metadata = detect_model_metadata(request.model_id, request.filename)
    
    # 2. Scanner Determinístico (Hugging Face / Cache)
    meta_service = ModelMetadata("model")
    await meta_service.resolve(repo_id=request.model_id, filename=request.filename)
    
    if meta_service._raw_metadata:
        # Mescla os dados reais (sobrescrevendo heurísticas se necessário)
        metadata["architecture"] = meta_service.architecture
        metadata["num_layers"] = meta_service.block_count
        metadata["context_length"] = meta_service.context_length
        metadata["file_size_gb"] = meta_service.file_size_gb
        
    # Sincroniza model_type com is_vision
    if metadata.get("model_type") == "vision":
        metadata["is_vision"] = True
    elif metadata.get("is_vision") is True:
        metadata["model_type"] = "vision"

    return {
        "status": "success",
        "detected_metadata": metadata
    }

@router.post("/download")
async def download_gguf(request: DownloadRequest):
    # Pre-validation for vision models
    if request.model_type == "vision" or (request.is_vision if request.is_vision is not None else False):
        if not request.mmproj_id or not request.mmproj_filename:
            raise HTTPException(status_code=400, detail="mmproj_id e mmproj_filename são obrigatórios para modelos de visão")

    # [ALERTA] Inserção imediata na base antes de iniciar o stream para garantir persistência 
    auto_meta = detect_model_metadata(request.model_id, request.filename)
    final_template = request.template if request.template else auto_meta.get("template", "chatml")
    final_reasoning = request.force_reasoning if request.force_reasoning is not None else auto_meta.get("force_reasoning", False)
    
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
        "kv_cache_quantization": request.kv_cache_quantization or "FP16",
        "num_layers": request.num_layers,
        "force_reasoning": final_reasoning,
        "flash_attn": request.flash_attn if request.flash_attn is not None else True,
        "is_vision": request.is_vision if request.is_vision is not None else (request.model_type == "vision"),
        "n_seq_max": request.n_seq_max or 1,
        "offload_kqv": request.offload_kqv if request.offload_kqv is not None else True,
        "kv_unified": request.kv_unified if request.kv_unified is not None else True,
        "logits_all": request.logits_all if request.logits_all is not None else False,
        "vision_on_cpu": request.vision_on_cpu if request.vision_on_cpu is not None else False,
        "file_path": None, # Pendente download
        "mmproj_path": None
    }
    save_models_db(db)
    logger.info(f"Model {request.model_alias} registered in DB (pending download)")

    async def progress_generator():
        logger.info(f"Progress generator started for {request.model_alias}")
        try:
            # Multiplos yields iniciais para garantir que o buffer de proxies/navegador seja liberado
            yield f"data: {json.dumps({'status': 'initializing', 'message': 'Conectando ao serviço de download...'})}\n\n"
            yield f"data: {json.dumps({'status': 'starting', 'message': f'Iniciando download de {request.model_alias}'})}\n\n"
            
            # Download the main file
            loop = asyncio.get_event_loop()
            file_path = await loop.run_in_executor(None, lambda: hf_hub_download(
                repo_id=request.model_id, 
                filename=request.filename,
                cache_dir=CACHE_DIR
            ))
            
            # Atualiza caminho principal no banco
            curr_db = load_models_db()
            if request.model_alias in curr_db:
                curr_db[request.model_alias]["file_path"] = file_path
                save_models_db(curr_db)
            
            yield f"data: {json.dumps({'status': 'downloading', 'message': 'Arquivo principal concluído. Baixando mmproj...'})}\n\n"

            mmproj_path = None
            if request.model_type == "vision" or (request.is_vision if request.is_vision is not None else False):
                mmproj_path = await loop.run_in_executor(None, lambda: hf_hub_download(
                    repo_id=request.mmproj_id, 
                    filename=request.mmproj_filename,
                    cache_dir=CACHE_DIR
                ))
                # Atualiza mmproj no banco
                curr_db = load_models_db()
                if request.model_alias in curr_db:
                    curr_db[request.model_alias]["mmproj_path"] = mmproj_path
                    save_models_db(curr_db)

            final_data = load_models_db().get(request.model_alias, {})
            yield f"data: {json.dumps({'status': 'success', 'message': 'Download concluído!', 'metadata': final_data})}\n\n"
        except Exception as e:
            logger.error(f"Download failed: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        progress_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/files/{repo_id:path}")
async def list_hf_files(repo_id: str):
    """Lista todos os arquivos de um repositório no Hugging Face."""
    try:
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, lambda: list_repo_files(repo_id))
        return {"status": "success", "repo_id": repo_id, "files": files}
    except Exception as e:
        logger.error(f"Error listing files for {repo_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Erro ao listar arquivos do repositório {repo_id}: {str(e)}")

@router.get("/")
async def list_gguf_models():
    db = load_models_db()
    updated = False
    
    # Auto-repair paths from HF cache if missing
    for alias, model in db.items():
        if alias in ['xtts', 'faster-whisper']: continue
        
        # 1. Checa arquivo principal
        if not model.get("file_path") and model.get("model_id") and model.get("filename"):
            try:
                # local_files_only=True apenas consulta o cache
                path = hf_hub_download(repo_id=model["model_id"], filename=model["filename"], local_files_only=True)
                if path and os.path.exists(path):
                    model["file_path"] = path
                    updated = True
                    logger.info(f"Auto-repaired path for {alias}: {path}")
            except:
                pass
        
        # 2. Checa mmproj (visão)
        if model.get("is_vision") and not model.get("mmproj_path") and model.get("mmproj_id") and model.get("mmproj_filename"):
            try:
                path = hf_hub_download(repo_id=model["mmproj_id"], filename=model["mmproj_filename"], local_files_only=True)
                if path and os.path.exists(path):
                    model["mmproj_path"] = path
                    updated = True
                    logger.info(f"Auto-repaired mmproj path for {alias}: {path}")
            except:
                pass
                
    if updated:
        save_models_db(db)
        
    return {"models": list(db.values())}

@router.put("/{model_alias}")
async def update_model_metadata(model_alias: str, request: UpdateMetadataRequest):
    from ..orchestrator import orchestrator
    
    db = load_models_db()
    if model_alias not in db:
        raise HTTPException(status_code=404, detail=f"Model {model_alias} not found")
        
    model_data = db[model_alias]
    
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        model_data[key] = value
        
    db[model_alias] = model_data
    save_models_db(db)
    
    # Descarrega o modelo da GPU se ele estiver carregado, para que a próxima chamada use as novas configs
    await orchestrator.unload_model(model_alias)
    
    return {"status": "success", "model": model_data}

@router.delete("/gguf/{model_alias}")
async def delete_model(model_alias: str):
    from ..orchestrator import orchestrator
    db = load_models_db()
    if model_alias in db:
        # Descarrega o modelo da GPU antes de remover do banco
        await orchestrator.unload_model(model_alias)
        
        del db[model_alias]
        save_models_db(db)
        return {"status": "success", "message": f"Model {model_alias} removed from database"}
    raise HTTPException(status_code=404, detail="Model not found")

@router.post("/warmup")
async def model_warmup(request: WarmupRequest):
    """Pré-carrega modelos na VRAM/RAM baseado nos tipos de tarefa."""
    from ..model_loader import model_loader
    from ..services.models_service import model_service

    loaded_models = []
    errors = []

    for item in request.models:
        task = item.task_type.lower()
        model_alias = item.model
        try:
            backend = item.backend

            if not backend:
                engine_settings = model_service.get_setting(task) or {}
                if isinstance(engine_settings, list):
                    cfg = next((p for p in engine_settings if p.get("alias") == model_alias or p.get("model") == model_alias), engine_settings[0] if engine_settings else {})
                else:
                    cfg = engine_settings

                if cfg.get("backend") == "api" or cfg.get("model_type") == "api":
                    backend = "api"
                elif cfg.get("num_layers") == 0:
                    backend = "cpu"
                else:
                    backend = cfg.get("backend", "gpu")

            adapter = model_loader.get(task, backend=backend, model_tag=model_alias)

            if hasattr(adapter, "load"):
                adapter.load()

            loaded_models.append({"task": task, "model": model_alias, "backend": backend})
            logger.info(f"Warmup: model '{model_alias}' loaded for task '{task}' on backend '{backend}'")

        except Exception as e:
            logger.error(f"Warmup: Error loading '{model_alias}' for task '{task}': {e}")
            errors.append(f"{task}/{model_alias}: {str(e)}")

    return {
        "status": "success" if not errors else "partial_success",
        "loaded": loaded_models,
        "errors": errors
    }
