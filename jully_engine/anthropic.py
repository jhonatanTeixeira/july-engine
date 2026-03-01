from fastapi import APIRouter, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from fastapi.responses import Response, StreamingResponse
from .bridge import bridge

router = APIRouter()

class MessageRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    
    # Allow extra fields for parity
    model_config = {"extra": "allow"}

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str

@router.post("/messages")
async def create_message(request: MessageRequest, http_request: Request):
    payload = request.model_dump()
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    payload['headers'] = headers
    
    response = await bridge.process_anthropic_message(payload)
    if isinstance(response, AsyncGenerator):
        return StreamingResponse(response, media_type="text/event-stream")
    return response

@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    embeddings = await bridge.process_embeddings(request.input, request.model, headers)
    data = [{"index": i, "embedding": emb} for i, emb in enumerate(embeddings)]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

@router.post("/audio/speech")
async def create_speech(request: SpeechRequest, http_request: Request):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    audio_bytes = await bridge.process_tts(request.input, request.voice, headers)
    return Response(content=audio_bytes, media_type="audio/mpeg")

@router.post("/audio/transcriptions")
async def create_transcription(
    http_request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    audio_bytes = await file.read()
    transcription = await bridge.process_stt(audio_bytes, language, headers)
    return {"text": transcription}
