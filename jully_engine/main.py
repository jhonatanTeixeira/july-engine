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
from .routers.openai import router as openai_router
from .routers.anthropic import router as anthropic_router
from .routers.models import router as models_router
from .routers.calculator import router as calculator_router
from .routers.monitoring import router as monitoring_router
from .routers.voice import router as voice_router
from .routers.search import router as search_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the bridge which starts all orchestrators
    await bridge.start()
    yield
    # Shutdown: Stop the bridge which stops all orchestrators
    await bridge.stop()

description = """
**July Engine** is a high-performance multimodal inference engine designed for hybrid operation.

### Core Capabilities:
* **Chat & Vision:** Compatible with OpenAI and Anthropic formats.
* **Text-to-Speech (TTS):** Using XTTS2 (cloning) and Piper (fast ONNX).
* **Speech-to-Text (STT):** Powered by FasterWhisper.
* **Image Editing & Generation:** Using InstructPix2Pix and Stable Diffusion.
* **Embeddings:** High-performance local vector generation.

### How to Use:
The engine routes requests based on custom HTTP headers:
1. **`x-backend`**: Set to `cpu`, `gpu`, or `api` to choose where the task runs.
2. **`x-base-url`**: When using the `api` backend, this specifies the external provider URL.
3. **`Authorization`**: Pass a Bearer token to be forwarded to external API providers.

All endpoints follow standard industry schemas for seamless integration with existing AI tools.
"""

app = FastAPI(
    title="July Engine", 
    version="2.0.0", 
    description=description,
    lifespan=lifespan,
    contact={
        "name": "July ecosystem Support",
    }
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router, prefix="/v1/openai")
app.include_router(anthropic_router, prefix="/v1/anthropic")
app.include_router(models_router)
app.include_router(calculator_router)
app.include_router(monitoring_router)
app.include_router(voice_router)
app.include_router(search_router)


@app.get("/health", tags=["July"])
async def health():
    return {"status": "online", "engine": "July Engine"}


if __name__ == "__main__":
    uvicorn.run("jully_engine.main:app", host="0.0.0.0", port=8000, reload=True)
