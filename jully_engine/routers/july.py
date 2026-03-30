import os
import time
import uuid
import base64
from typing import List, Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Assumindo que o bridge já está importado ou acessível
from ..bridge import bridge

router = APIRouter(prefix="/july/v1", tags=["July Custom"])


async def save_upload_stream(upload_file: UploadFile, dest_folder: str = "storage/temp") -> str:
    """Lê o arquivo binário em pedaços e salva no disco sem inflar a RAM."""
    os.makedirs(dest_folder, exist_ok=True)
    
    # Gera um nome de arquivo único para evitar colisão de requests paralelos
    ext = upload_file.filename.split('.')[-1] if '.' in upload_file.filename else 'bin'
    file_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(dest_folder, file_name)
    
    # Lendo em chunks de 1MB
    chunk_size = 1024 * 1024 
    
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)
            
    return file_path


@router.post("/vision/video/describe")
async def describe_video(
    http_request: Request,
    file: UploadFile = File(...),
    interval_sec: Optional[float] = Form(2.0), # Deixa o cliente escolher a densidade!
    frames_per_grid: Optional[int] = Form(4),  # Quantos frames por lote
    model: Optional[str] = Form(None),
    strategy: Optional[str] = Form("default"), # Pode ser "default", "interaction" ou "emotion"
    description_model: Optional[str] = Form(None),
):
    """
    Analisa os frames visuais de um vídeo e retorna uma descrição detalhada 
    das ações, ambiente e pessoas. (Não transcreve áudio).
    """
    headers = dict(http_request.headers)
    
    # Salva o vídeo via stream (protegendo a RAM)
    saved_video_path = await save_upload_stream(file)
    
    payload = {
        "video_path": saved_video_path,
        "interval_sec": interval_sec,
        "frames_per_grid": frames_per_grid,
        "model": model,
        "strategy": strategy,
        "description_model": description_model,
    }
    
    try:
        # Agora o nome deixa claro que vamos invocar o VLM (Olhos), e não o STT (Ouvidos)
        result = await bridge.process_video_description(payload, headers)
        
        return JSONResponse(content={"visual_narrative": result})
    finally:
        if os.path.exists(saved_video_path):
            os.remove(saved_video_path)


@router.post("/vision/faces/extract")
async def extract_faces(
    http_request: Request,
    files: List[UploadFile] = File(...), # Recebe N imagens num único POST
    model: Optional[str] = Form(None)
):
    headers = dict(http_request.headers)
    
    images_b64 = []
    for file in files:
        bytes_data = await file.read()
        b64_str = base64.b64encode(bytes_data).decode('utf-8')
        images_b64.append(b64_str)
        
    payload = {
        "images": images_b64,
        "model": model
    }
    
    description = await bridge.process_face_extraction(payload, headers)
    
    return JSONResponse(content={"faces_description": description})


@router.post("/vision/images/describe")
async def describe_images(
    http_request: Request,
    files: List[UploadFile] = File(...), # Aceita array de imagens
    prompt: str = Form("Describe these images in detail."), # Prompt customizável
    model: Optional[str] = Form(None)
):
    headers = dict(http_request.headers)
    
    images_b64 = []
    for file in files:
        bytes_data = await file.read()
        b64_str = base64.b64encode(bytes_data).decode('utf-8')
        images_b64.append(b64_str)
        
    payload = {
        "images": images_b64,
        "prompt": prompt,
        "model": model
    }
    
    # Chama o Bridge (que fará o repasse para o orquestrador VLM)
    descriptions = await bridge.process_image_description(payload, headers)
    
    # Devolvemos uma lista com a descrição de cada imagem, ou um consolidado
    return JSONResponse(content={"descriptions": descriptions})