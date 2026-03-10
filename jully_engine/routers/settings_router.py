from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ..services.settings_service import SettingsService

router = APIRouter(prefix="/v1/settings", tags=["Settings"])
service = SettingsService()

@router.get("/")
def get_all_settings():
    return service.get_all()

@router.post("/")
def save_settings(payload: Dict[str, Any]):
    for key, value in payload.items():
        service.set(key, value)
    return {"success": True}
