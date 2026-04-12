import os
import logging
from dotenv import load_dotenv

# --- CONFIGURAÇÃO DE LOGS GLOBAL (COLORIDO) ---
class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Configura o logger raiz para o ecossistema JulyEngine
root_logger = logging.getLogger("JulyEngine")
root_logger.setLevel(logging.DEBUG if os.environ.get("DEBUG") == "true" else logging.INFO)
root_logger.propagate = False # Evita duplicidade se o root do sistema também estiver configurado

if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    root_logger.addHandler(console_handler)

# Silenciar logs barulhentos de bibliotecas externas
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_cpp").setLevel(logging.WARNING)

# --- FIM CONFIGURAÇÃO DE LOGS ---

env = os.environ['ENV'] if 'ENV' in os.environ else None
load_dotenv(f'.env.{env}' if env else '.env', verbose=True)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .bridge import bridge
from .routers.openai import router as openai_router
from .routers.anthropic import router as anthropic_router
from .routers.models import router as models_router
from .routers.calculator import router as calculator_router
from .routers.monitoring import router as monitoring_router
from .routers.voice import router as voice_router
from .routers.search import router as search_router
from .routers.july import router as july_router

from .routers.settings_router import router as settings_router
from .routers.mcps_router import router as mcps_router
from .routers.webhooks_router import router as webhooks_router
from .services.external_mcp import external_mcp_manager
from .events import event_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the bridge which starts all orchestrators
    event_manager.start()
    await bridge.start()
    await external_mcp_manager.start()
    yield
    # Shutdown: Stop the bridge which stops all orchestrators
    await external_mcp_manager.stop()
    await bridge.stop()
    event_manager.stop()

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
app.include_router(settings_router)
app.include_router(mcps_router)
app.include_router(webhooks_router)
app.include_router(july_router)


@app.get("/health", tags=["July"])
async def health():
    return {"status": "online", "engine": "July Engine"}


if __name__ == "__main__":
    uvicorn.run("jully_engine.main:app", host="0.0.0.0", port=8000, reload=True)
