from fastapi import APIRouter, HTTPException
import os
import json
import logging
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

logger = logging.getLogger("JulyEngine.Routers.Models")

router = APIRouter(prefix="/models/gguf", tags=["Models"])

# Define cache dir and models.json path
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
MODELS_JSON_PATH = os.path.join(CACHE_DIR, "july_models.json")

def load_models_db() -> Dict[str, Any]:
    if os.path.exists(MODELS_JSON_PATH):
        try:
            with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {MODELS_JSON_PATH}: {e}")
            return {}
    return {}

def save_models_db(db: Dict[str, Any]):
    os.makedirs(os.path.dirname(MODELS_JSON_PATH), exist_ok=True)
    with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)

class DownloadRequest(BaseModel):
    model_alias: str
    model_type: str # "text" or "vision"
    model_id: str # Hugging Face repo id
    filename: str
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    template: Optional[str] = None

class UpdateMetadataRequest(BaseModel):
    model_type: Optional[str] = None
    model_id: Optional[str] = None
    filename: Optional[str] = None
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    template: Optional[str] = None

@router.post("/download")
async def download_gguf(request: DownloadRequest):
    try:
        if request.model_type == "vision" and (not request.mmproj_id or not request.mmproj_filename):
            raise HTTPException(status_code=400, detail="mmproj_id and mmproj_filename are required for vision models")

        # Download the main file
        file_path = hf_hub_download(repo_id=request.model_id, filename=request.filename)
        
        # Download mmproj if it's a vision model
        mmproj_path = None
        if request.model_type == "vision":
            mmproj_path = hf_hub_download(repo_id=request.mmproj_id, filename=request.mmproj_filename)

        db = load_models_db()
        db[request.model_alias] = {
            "model_alias": request.model_alias,
            "model_type": request.model_type,
            "model_id": request.model_id,
            "filename": request.filename,
            "mmproj_id": request.mmproj_id,
            "mmproj_filename": request.mmproj_filename,
            "template": request.template,
            "file_path": file_path,
            "mmproj_path": mmproj_path
        }
        save_models_db(db)

        return {"status": "success", "message": f"Model {request.model_alias} downloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to download GGUF from {request.model_id}/{request.filename}: {e}")
        return {"status": "error", "message": str(e)}

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
