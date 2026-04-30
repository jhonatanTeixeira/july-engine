from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from ..services.models_service import ModelsService

router = APIRouter(prefix="/v1/models", tags=["Models"])
service = ModelsService()

@router.get("/", response_model=List[Dict[str, Any]])
def get_all_models():
    return service.get_all()

@router.post("/")
async def create_or_update_model(payload: Dict[str, Any]):
    from ..orchestrators.gpu_orchestrator import gpu_orchestrator
    model_alias = payload.get("model_alias")
    if not model_alias:
        raise HTTPException(status_code=400, detail="model_alias is required")
    service.set(model_alias, payload)
    
    # Descarrega o modelo da GPU se ele estiver carregado
    await gpu_orchestrator.unload_model(model_alias)
    
    return {"success": True, "model_alias": model_alias}

@router.get("/{model_alias}")
def get_model(model_alias: str):
    model = service.get(model_alias)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.delete("/{model_alias}")
async def delete_model(model_alias: str):
    from ..orchestrators.gpu_orchestrator import gpu_orchestrator
    # Descarrega o modelo da GPU antes de remover
    await gpu_orchestrator.unload_model(model_alias)
    
    success = service.delete(model_alias)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"success": True}
