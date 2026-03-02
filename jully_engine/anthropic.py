from fastapi import APIRouter, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
import time
import base64
from fastapi.responses import Response, StreamingResponse
from .bridge import bridge

router = APIRouter(tags=["Anthropic"])

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

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "pix2pix"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"

@router.post("/messages")
async def create_message(request: MessageRequest, http_request: Request):
    payload = request.model_dump()
    headers = dict(http_request.headers)
    
    response = await bridge.process_anthropic_message(payload, headers)
    if isinstance(response, AsyncGenerator):
        return StreamingResponse(response, media_type="text/event-stream")
    return response

@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    embeddings = await bridge.process_embeddings(payload, headers)
    data = [{"index": i, "embedding": emb} for i, emb in enumerate(embeddings)]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

@router.post("/audio/speech")
async def create_speech(request: SpeechRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    # Note: Anthropic doesn't have a standard speech endpoint, 
    # but we provide parity with OpenAI one here.
    output_path = await bridge.process_tts(payload, headers)
    
    import os
    if output_path and os.path.exists(output_path):
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        return Response(content=audio_bytes, media_type="audio/wav")
    
    return Response(status_code=500, content="TTS failed")

@router.post("/audio/transcriptions")
async def create_transcription(
    http_request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
):
    headers = dict(http_request.headers)
    audio_bytes = await file.read()
    payload = {
        "audio": audio_bytes,
        "model": model,
        "language": language
    }
    transcription = await bridge.process_stt(payload, headers)
    return {"text": transcription}

@router.post("/images/generations")
async def create_image_generation(request: ImageGenerationRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    image_base64 = await bridge.process_image_generation(payload, headers)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": image_base64}]
    }

@router.post("/images/edits")
async def create_image_edit(
    http_request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
):
    headers = dict(http_request.headers)
    image_bytes = await image.read()
    image_data = base64.b64encode(image_bytes).decode()
    payload = {
        "image": image_data,
        "prompt": prompt,
        "model": model
    }
    edited_image_base64 = await bridge.process_image_edit(payload, headers)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": edited_image_base64}]
    }
