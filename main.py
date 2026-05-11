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

env = os.environ['ENV'] if 'ENV' in os.environ else None
load_dotenv(f'.env.{env}' if env else '.env', verbose=True)

# Configura o logger raiz para o ecossistema JulyEngine
root_logger = logging.getLogger("JulyEngine")
root_logger.setLevel(logging.DEBUG if os.environ.get("DEBUG") == "true" else logging.INFO)
root_logger.propagate = False # Evita duplicidade se o root do sistema também estiver configurado

if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    root_logger.addHandler(console_handler)

# Silenciar logs barulhentos de bibliotecas externas
logging.getLogger("uvicorn.access").setLevel(logging.DEBUG if os.environ.get("DEBUG") == "true" else logging.WARNING)
logging.getLogger("httpx").setLevel(logging.DEBUG if os.environ.get("DEBUG") == "true" else logging.WARNING)
logging.getLogger("llama_cpp").setLevel(logging.DEBUG if os.environ.get("DEBUG") == "true" else logging.WARNING)

# --- FIM CONFIGURAÇÃO DE LOGS ---


import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.bridge import bridge
from app.routers.openai import router as openai_router
from app.routers.anthropic import router as anthropic_router
from app.routers.models import router as models_router
from app.routers.calculator import router as calculator_router
from app.routers.monitoring import router as monitoring_router
from app.routers.voice import router as voice_router
from app.routers.search import router as search_router
from app.routers.july import router as july_router

from app.routers.settings_router import router as settings_router
from app.routers.services_router import router as services_router
from app.routers.mcps_router import router as mcps_router
from app.routers.webhooks_router import router as webhooks_router
from app.services.external_mcp import external_mcp_manager
from app.events import event_manager
from fastapi.staticfiles import StaticFiles
import uuid
from app.context import request_id_var, acquired_instances_var

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

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Gera um ID único para a requisição HTTP
    rid = str(uuid.uuid4())
    token_rid = request_id_var.set(rid)
    # Inicializa o registro de instâncias para esta requisição
    token_instances = acquired_instances_var.set({})
    
    try:
        from fastapi.responses import StreamingResponse
        
        # Função de limpeza que aceita um dicionário explícito
        def cleanup_explicit(to_cleanup):
            if to_cleanup:
                for pool, inst in to_cleanup.items():
                    try:
                        pool._force_release(inst)
                    except:
                        pass
        
        try:
            response = await call_next(request)
            
            # Captura as instâncias adquiridas ATÉ AGORA nesta request
            current_acquired = acquired_instances_var.get().copy()
            
            if isinstance(response, StreamingResponse):
                original_iterator = response.body_iterator
                
                async def wrapped_iterator():
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    finally:
                        cleanup_explicit(current_acquired)
                
                response.body_iterator = wrapped_iterator()
                return response
                
            # Para respostas síncronas
            cleanup_explicit(current_acquired)
            return response
        finally:
            # Se deu erro antes de 'current_acquired' ser definido ou antes da resposta
            if 'current_acquired' not in locals():
                cleanup_explicit(acquired_instances_var.get())
            elif not isinstance(response, StreamingResponse):
                # Já limpamos acima para sync, mas por segurança...
                pass
    finally:
        # Limpa o contexto
        request_id_var.reset(token_rid)
        acquired_instances_var.reset(token_instances)

# Servir arquivos estáticos do diretório storage/voices
storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "voices")
app.mount("/storage", StaticFiles(directory=storage_path), name="storage")

app.include_router(openai_router, prefix="/v1/openai")
app.include_router(anthropic_router, prefix="/v1/anthropic")
app.include_router(models_router)
app.include_router(calculator_router)
app.include_router(monitoring_router)
app.include_router(voice_router)
app.include_router(search_router)
app.include_router(settings_router)
app.include_router(services_router)
app.include_router(mcps_router)
app.include_router(webhooks_router)
app.include_router(july_router)


@app.get("/health", tags=["July"])
async def health():
    return {"status": "online", "engine": "July Engine"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
