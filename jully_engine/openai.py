from fastapi import APIRouter, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
import time
import base64
import os
from fastapi.responses import Response, StreamingResponse
from .bridge import bridge

router = APIRouter()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    
    # Allow extra fields for num_ctx or others (from extra_body)
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

# --- Response DTOs for Swagger Documentation ---

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., examples=["chatcmpl-123"])
    object: str = "chat.completion"
    created: int = Field(..., examples=[1677652288])
    model: str = Field(..., examples=["qwen3-0.6b.gguf"])
    choices: List[Dict[str, Any]] = Field(..., examples=[{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help you?"},
        "finish_reason": "stop"
    }])
    usage: Dict[str, int] = Field(..., examples=[{"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}])

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]] = Field(..., examples=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}])
    model: str = Field(..., examples=["bge-micro"])
    usage: Dict[str, int] = Field(..., examples=[{"prompt_tokens": 8, "total_tokens": 8}])

class ImageResponse(BaseModel):
    created: int = Field(..., examples=[1677652288])
    data: List[Dict[str, str]] = Field(..., examples=[{"b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."}])

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    payload = request.model_dump()
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    payload['headers'] = headers
    
    response = await bridge.process_openai_chat(payload)

    if isinstance(response, AsyncGenerator):
        return StreamingResponse(response, media_type="text/event-stream")

    return response

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    embeddings = await bridge.process_embeddings(request.input, request.model, headers)
    data = [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embeddings)]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }

@router.post("/audio/speech")
async def create_speech(request: SpeechRequest, http_request: Request):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    output_path = await bridge.process_tts(request.input, request.voice, request.model, headers)
    
    if output_path and os.path.exists(output_path):
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        return Response(content=audio_bytes, media_type="audio/wav")
    
    return Response(status_code=500, content="TTS failed to generate audio")

@router.post("/audio/transcriptions")
async def create_transcription(
    http_request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    audio_bytes = await file.read()
    transcription = await bridge.process_stt(audio_bytes, model, language, headers)
    return {"text": transcription}

@router.post("/images/edits", response_model=ImageResponse)
async def create_image_edit(
    http_request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith('x-')}
    image_bytes = await image.read()
    image_data = base64.b64encode(image_bytes).decode()
    edited_image_base64 = await bridge.process_image_edit(image_data, prompt, model, headers)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": edited_image_base64}]
    }

@router.post("/images/generations", response_model=ImageResponse)
async def create_image_generation(request: ImageGenerationRequest, http_request: Request):
    headers = {k: v for k, v in http_request.headers.items() if k.startswith("x-")}
    image_base64 = await bridge.process_image_generation(request.prompt, request.model, headers)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": image_base64}]
    }
