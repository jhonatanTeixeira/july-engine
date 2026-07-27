import os
import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger("JulyEngine.Routers.Models")

router = APIRouter(prefix="/models/gguf", tags=["Models"])

# Define cache dir and models.json path
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
CACHE_DIR = os.path.join(CACHE_DIR, 'hub')
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


# model_alias is used as a path segment in /models/gguf/{alias} routes — a "/"
# here (e.g. someone pasting the HF repo id instead of a short name) breaks
# routing for that model going forward.
def _validate_alias_no_slash(v: Optional[str]) -> Optional[str]:
    if v and "/" in v:
        raise ValueError("model_alias cannot contain '/' — use a short name, not the HF repo id")
    return v


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
    is_audio: Optional[bool] = None  # native audio via the same mmproj as is_vision — see llama_gguf.py
    flash_attn: Optional[bool] = True
    n_seq_max: Optional[int] = 1
    offload_kqv: Optional[bool] = True
    kv_unified: Optional[bool] = True
    logits_all: Optional[bool] = False
    vision_on_cpu: Optional[bool] = False
    cpu_moe: Optional[bool] = False
    n_cpu_moe: Optional[int] = 0

    _alias_no_slash = field_validator("model_alias")(_validate_alias_no_slash)

class UpdateMetadataRequest(BaseModel):
    model_alias: Optional[str] = None  # rename target; None/unset keeps the existing alias
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
    is_audio: Optional[bool] = None
    flash_attn: Optional[bool] = None
    n_seq_max: Optional[int] = None
    offload_kqv: Optional[bool] = None
    kv_unified: Optional[bool] = None
    logits_all: Optional[bool] = None
    vision_on_cpu: Optional[bool] = None
    cpu_moe: Optional[bool] = None
    n_cpu_moe: Optional[int] = None
    file_path: Optional[str] = None
    mmproj_path: Optional[str] = None

    _alias_no_slash = field_validator("model_alias")(_validate_alias_no_slash)

class WarmupItem(BaseModel):
    task_type: str  # text_chat, vision_chat, tts, stt, embeddings, image_generation, etc.
    model: str  # model alias/tag
    backend: Optional[str] = None  # gpu, cpu — se omitido, resolve via config

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
    from llama_gguf.resource_calculator import ModelMetadata

    # 1. Heurísticas baseadas em nome (Regex)
    metadata = detect_model_metadata(request.model_id, request.filename)

    # 2. Scanner Determinístico (Hugging Face cache — só lê o header real se o
    # arquivo já estiver em cache local; caso contrário resolve_remote() abaixo
    # só consegue obter o tamanho do arquivo via HEAD, não os campos do GGUF).
    model_path = "model"
    try:
        cached_path = hf_hub_download(repo_id=request.model_id, filename=request.filename, local_files_only=True)
        if cached_path and os.path.exists(cached_path):
            model_path = cached_path
    except Exception:
        pass

    meta_service = ModelMetadata(model_path, repo_id=request.model_id, filename=request.filename)
    if not os.path.exists(model_path):
        await meta_service.resolve_remote()

    if meta_service.data:
        # Mescla os dados reais (sobrescrevendo heurísticas se necessário)
        metadata["architecture"] = meta_service.architecture
        metadata["num_layers"] = meta_service.block_count
        metadata["context_length"] = meta_service.context_length
    metadata["file_size_gb"] = round(meta_service.file_size_gb, 4)

    # Sincroniza model_type com is_vision
    if metadata.get("model_type") == "vision":
        metadata["is_vision"] = True
    elif metadata.get("is_vision") is True:
        metadata["model_type"] = "vision"

    return {
        "status": "success",
        "detected_metadata": metadata
    }

async def _download_model_files(model_alias: str) -> Dict[str, Any]:
    # hf_hub_download is a cheap cache lookup (near-instant) once the file is
    # already on disk, so this is safe to re-run any time a model is saved or
    # (re)downloaded — it only actually fetches anything the first time, or
    # after the cache was evicted. Reads model_id/filename from the model's
    # current DB row rather than a request payload, so it reflects whatever
    # was last saved, not stale/unsaved form state.
    db = load_models_db()
    row = db.get(model_alias)
    if not row:
        raise HTTPException(status_code=404, detail=f"Model {model_alias} not found")

    loop = asyncio.get_event_loop()
    file_path = await loop.run_in_executor(None, lambda: hf_hub_download(
        repo_id=row["model_id"],
        filename=row["filename"],
    ))
    curr_db = load_models_db()
    if model_alias in curr_db:
        curr_db[model_alias]["file_path"] = file_path
        save_models_db(curr_db)

    if (row.get("model_type") in ("vision", "audio") or row.get("is_vision") or row.get("is_audio")) and row.get("mmproj_id") and row.get("mmproj_filename"):
        mmproj_path = await loop.run_in_executor(None, lambda: hf_hub_download(
            repo_id=row["mmproj_id"],
            filename=row["mmproj_filename"],
        ))
        curr_db = load_models_db()
        if model_alias in curr_db:
            curr_db[model_alias]["mmproj_path"] = mmproj_path
            save_models_db(curr_db)

    return load_models_db().get(model_alias, {})


async def _download_progress_stream(model_alias: str):
    logger.info(f"Progress generator started for {model_alias}")
    try:
        # Multiplos yields iniciais para garantir que o buffer de proxies/navegador seja liberado
        yield f"data: {json.dumps({'status': 'initializing', 'message': 'Conectando ao serviço de download...'})}\n\n"
        yield f"data: {json.dumps({'status': 'starting', 'message': f'Iniciando download de {model_alias}'})}\n\n"
        yield f"data: {json.dumps({'status': 'downloading', 'message': 'Baixando arquivos...'})}\n\n"

        final_data = await _download_model_files(model_alias)
        yield f"data: {json.dumps({'status': 'success', 'message': 'Download concluído!', 'metadata': final_data})}\n\n"
    except Exception as e:
        logger.error(f"Download failed: {e}")
        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"


def _stream_download_response(model_alias: str) -> StreamingResponse:
    return StreamingResponse(
        _download_progress_stream(model_alias),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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

    # model_type is legacy/informational — the admin form drives capability via the
    # is_vision/is_audio toggles only, so derive model_type FROM them here rather
    # than trusting request.model_type (which the form always submits as "text",
    # having no field of its own). This is what runtime handler selection in
    # llama_gguf.py used to silently miss when model_type disagreed with is_vision.
    final_is_vision = request.is_vision if request.is_vision is not None else (request.model_type == "vision")
    final_is_audio = request.is_audio if request.is_audio is not None else (request.model_type == "audio")
    final_model_type = "vision" if final_is_vision else "audio" if final_is_audio else (request.model_type or "text")

    db = load_models_db()
    db[request.model_alias] = {
        "model_alias": request.model_alias,
        "model_type": final_model_type,
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
        "is_vision": final_is_vision,
        "is_audio": final_is_audio,
        "n_seq_max": request.n_seq_max or 1,
        "offload_kqv": request.offload_kqv if request.offload_kqv is not None else True,
        "kv_unified": request.kv_unified if request.kv_unified is not None else True,
        "logits_all": request.logits_all if request.logits_all is not None else False,
        "vision_on_cpu": request.vision_on_cpu if request.vision_on_cpu is not None else False,
        "cpu_moe": request.cpu_moe if request.cpu_moe is not None else False,
        "n_cpu_moe": request.n_cpu_moe or 0,
        "file_path": None, # Pendente download
        "mmproj_path": None
    }
    save_models_db(db)
    logger.info(f"Model {request.model_alias} registered in DB (pending download)")

    return _stream_download_response(request.model_alias)


@router.post("/{model_alias}/download")
async def redownload_model(model_alias: str):
    # Download-only: no metadata is written here (that's what PUT /{model_alias}
    # or POST /download are for) — this just (re)ensures the already-saved
    # model_id/filename for `model_alias` is present on disk.
    if model_alias not in load_models_db():
        raise HTTPException(status_code=404, detail=f"Model {model_alias} not found")
    return _stream_download_response(model_alias)

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
        
        # 2. Checa mmproj (visão/áudio — mesmo arquivo mmproj para ambas capacidades)
        if (model.get("is_vision") or model.get("is_audio")) and not model.get("mmproj_path") and model.get("mmproj_id") and model.get("mmproj_filename"):
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
    new_alias = update_data.pop("model_alias", None)
    for key, value in update_data.items():
        model_data[key] = value

    # Re-sync model_type from is_vision/is_audio after the merge — the edit form
    # has no model_type field of its own, so a bare save (which explicitly sends
    # model_type=None) would otherwise stomp it back to None on every edit even
    # when is_vision/is_audio are true. See the same derivation in download_gguf.
    if model_data.get("is_vision"):
        model_data["model_type"] = "vision"
    elif model_data.get("is_audio"):
        model_data["model_type"] = "audio"
    elif not model_data.get("model_type"):
        model_data["model_type"] = "text"

    final_alias = model_alias
    if new_alias and new_alias != model_alias:
        if new_alias in db:
            raise HTTPException(status_code=409, detail=f"Model alias '{new_alias}' already in use")
        del db[model_alias]
        model_data["model_alias"] = new_alias
        final_alias = new_alias

    db[final_alias] = model_data
    save_models_db(db)

    # Descarrega o modelo da GPU se ele estiver carregado, para que a próxima chamada use as novas configs
    # (sob o alias antigo e, se houve rename, também sob o novo, por segurança).
    await orchestrator.unload_model(model_alias)
    if final_alias != model_alias:
        await orchestrator.unload_model(final_alias)

    # Ensures the file for the (possibly just-edited) model_id/filename is on
    # disk — a no-op fetch if it's already cached, a real download otherwise.
    # Errors here don't roll back the metadata save; re-downloading can be
    # retried via the "Baixar" button, which surfaces progress/errors directly.
    try:
        model_data = await _download_model_files(final_alias)
    except Exception as e:
        logger.error(f"Auto-download after save failed for {final_alias}: {e}")

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

                if cfg.get("num_layers") == 0:
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
