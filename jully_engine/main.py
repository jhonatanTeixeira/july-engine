import os
from dotenv import load_dotenv

env = os.environ['ENV'] if 'ENV' in os.environ else None
load_dotenv(f'.env.{env}' if env else '.env', verbose=True)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .bridge import bridge
from .resource_manager import resource_manager
from .openai import router as openai_router
from .anthropic import router as anthropic_router
from .voice_service import voice_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the bridge which starts all orchestrators
    await bridge.start()
    yield
    # Shutdown: Stop the bridge which stops all orchestrators
    await bridge.stop()

app = FastAPI(title="July Engine", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router, prefix="/v1/openai")
app.include_router(anthropic_router, prefix="/v1/anthropic")

@app.get("/health")
async def health():
    return {"status": "online", "engine": "July Engine"}

@app.get("/speech/voices")
async def list_voices():
    return voice_service.list_voices()

@app.post("/speech/voices")
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

@app.get("/status")
async def get_status():
    status_data = {
        "cpu": resource_manager.get_cpu_usage(),
        "ram": resource_manager.get_ram_usage(),
    }
    
    vram_info = resource_manager.get_vram_info()
    if vram_info:
        status_data["vram"] = {
            "total": vram_info["total"],
            "free": vram_info["free"],
            "used": vram_info["used"]
        }
    else:
        status_data["vram"] = "not_managed"

    return status_data

if __name__ == "__main__":
    uvicorn.run("jully_engine.main:app", host="0.0.0.0", port=8000, reload=True)
