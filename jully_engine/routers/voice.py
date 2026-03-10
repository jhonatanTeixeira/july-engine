from fastapi import APIRouter, HTTPException, Request
from ..bridge import bridge
from ..services import voice_service

router = APIRouter(prefix="/speech", tags=["Voice"])


@router.get("/voices")
async def list_voices():
    return voice_service.list_voices()


@router.post("/voices")
async def add_voice(request: Request):
    form = await request.form()
    name = form.get("name")
    language = form.get("language", "en")
    file = form.get("file")
    voice_type = form.get("type", "clone") # clone or piper
    
    if not name or not file:
        return {"error": "Missing name or file"}, 400
        
    content = await file.read()
    new_voice = voice_service.add_voice(name, language, content, voice_type)
    
    return new_voice
