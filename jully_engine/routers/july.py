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


@router.post("/vision/face/embedding")
async def get_face_embedding(
    http_request: Request,
    payload: dict
):
    """
    Recebe um crop de rosto em base64 e retorna o embedding ArcFace via DeepFace.
    Usado pelo jully_photos para matching facial delegado à Engine.
    """
    image_b64 = payload.get("image")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "Campo 'image' é obrigatório."})
    
    try:
        import io
        import numpy as np
        from PIL import Image as PILImage
        from deepface import DeepFace
        
        img_bytes = base64.b64decode(image_b64)
        img_pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        
        rep = DeepFace.represent(
            img_path=img_np,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True
        )
        embedding = rep[0]["embedding"]
        return JSONResponse(content={"embedding": embedding})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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


@router.post("/rag")
async def add_rag(
    http_request: Request,
    payload: dict
):
    """Adiciona um texto/descrição ao banco vetorial PGVector da Engine"""
    headers = dict(http_request.headers)
    backend = headers.get("x-backend", "api")
    
    model = payload.get("model", "bge-micro")
    text = payload.get("text")
    metadata = payload.get("metadata", {})
    collection = payload.get("collection", "july_memory")
    
    if not text:
        return JSONResponse(status_code=400, content={"error": "O campo 'text' é obrigatório no payload."})
        
    try:
        from ..domain.memory import Memory
        memory = Memory(backend=backend, model_tag=model)
        success = await memory.add_to_rag(text=text, metadata=metadata, collection=collection)
        return JSONResponse(content={"success": success, "collection": collection})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/rag")
async def search_rag(
    http_request: Request,
    query: str,
    collection: str = "july_memory",
    top_k: int = 3,
    model: str = "bge-micro"
):
    """Busca o contexto associado ao input num database MultiTenant/Segmentado"""
    headers = dict(http_request.headers)
    backend = headers.get("x-backend", "api")
    
    if not query:
        return JSONResponse(status_code=400, content={"error": "A querystring 'query' é obrigatória."})
        
    try:
        from ..domain.memory import Memory
        memory = Memory(backend=backend, model_tag=model)
        result = await memory.search(query=query, top_k=top_k, collection=collection)
        return JSONResponse(content={"results": result, "collection": collection})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/rag/vector")
async def add_rag_vector(
    http_request: Request,
    payload: dict
):
    """Adiciona um vetor matemático bruto (ex: Tracking de Rostos) com metadados."""
    headers = dict(http_request.headers)
    backend = headers.get("x-backend", "api")
    
    vector = payload.get("vector")
    metadata = payload.get("metadata", {})
    collection = payload.get("collection", "july_memory")
    text = payload.get("text", "")
    
    if not vector:
         return JSONResponse(status_code=400, content={"error": "O campo 'vector' é obrigatório."})
         
    try:
        from ..domain.memory import Memory
        memory = Memory(backend=backend, model_tag="bge-micro")
        success = await memory.add_vector_to_rag(embedding=vector, text=text, metadata=metadata, collection=collection)
        return JSONResponse(content={"success": success, "collection": collection})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rag/search")
async def search_rag_details(
    http_request: Request,
    payload: dict
):
    """Busca avançada MultiModal que retorna IDs, Distâncias e Metadados."""
    headers = dict(http_request.headers)
    backend = headers.get("x-backend", "api")
    
    query_text = payload.get("query", None)
    vector = payload.get("vector", None)
    collection = payload.get("collection", "july_memory")
    top_k = int(payload.get("top_k", 3))
    model = payload.get("model", "bge-micro")
    
    if not query_text and not vector:
         return JSONResponse(status_code=400, content={"error": "Envie 'query' (Texto) ou 'vector' (Matriz Float)."})
         
    try:
        from ..domain.memory import Memory
        memory = Memory(backend=backend, model_tag=model)
        
        if vector:
            result = await memory.search_with_details_vector(query_embedding=vector, top_k=top_k, collection=collection)
        else:
            emb_payload = {"input": query_text}
            embedding_result = await memory.embed(emb_payload)
            if embedding_result and len(embedding_result) > 0:
                emb = embedding_result[0] if isinstance(embedding_result[0], list) else embedding_result
                result = await memory.search_with_details_vector(query_embedding=emb, top_k=top_k, collection=collection)
            else:
                result = []
                
        return JSONResponse(content={"results": result, "collection": collection})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.put("/rag/update")
async def update_rag_embedding(
    http_request: Request,
    payload: dict
):
    """Substitui um Vetor Específico (usado para Tracking de Rosto)"""
    doc_id = payload.get("id")
    vector = payload.get("vector")
    
    if not doc_id or not vector:
         return JSONResponse(status_code=400, content={"error": "Forneça 'id' e 'vector'."})
         
    try:
        from ..domain.memory import Memory
        memory = Memory(backend="api", model_tag="bge-micro")
        await memory.update_embedding(doc_id=str(doc_id), new_embedding=vector)
        return JSONResponse(content={"success": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})