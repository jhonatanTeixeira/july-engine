from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import pydantic
from ..services.voice_service import voice_service

router = APIRouter(prefix="/v1/voices", tags=["Voices"])

class VoiceMetadataResponse(pydantic.BaseModel):
    id: str
    name: str
    language: str
    path: Optional[str] = None
    metadata: Optional[dict] = {}

@router.get("/", response_model=List[VoiceMetadataResponse])
async def list_voices():
    return voice_service.list_voices()

@router.post("/upload")
async def upload_voice(
    name: str = Form(...),
    language: str = Form(...),
    file: UploadFile = File(...)
):
    content = await file.read()
    try:
        new_voice = voice_service.add_voice(name, language, content)
        return new_voice
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{voice_id}")
async def delete_voice(voice_id: str):
    success = voice_service.delete_voice(voice_id)
    if not success:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"success": True}

class UpdateVoiceRequest(pydantic.BaseModel):
    name: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[dict] = None

@router.patch("/{voice_id}")
async def update_voice(voice_id: str, request: UpdateVoiceRequest):
    updated = voice_service.update_voice(
        voice_id, 
        name=request.name, 
        language=request.language, 
        metadata=request.metadata
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Voice not found")
    return updated

@router.post("/{voice_id}/clean")
async def clean_voice(voice_id: str):
    success = voice_service.clean_voice(voice_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clean voice audio or voice not found")
    return {"success": True}
