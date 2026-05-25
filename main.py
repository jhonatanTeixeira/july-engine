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

from july_telemetry.otel import setup_otel
setup_otel("july-engine")

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


import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
from app.bridge import bridge
from july_routers.openai import router as openai_router, set_bridge as openai_set_bridge
from july_routers.anthropic import router as anthropic_router, set_bridge as anthropic_set_bridge
from july_routers.search import router as search_router, set_bridge as search_set_bridge
from july_routers.july import router as july_router, set_bridge as july_set_bridge
from july_routers.calculator import router as calculator_router, set_bridge as calculator_set_bridge
from july_routers.services_router import router as services_router
from app.routers.models import router as models_router
from app.routers.monitoring import router as monitoring_router
from app.routers.voice import router as voice_router
from app.routers.settings_router import router as settings_router
from app.routers.webhooks_router import router as webhooks_router

openai_set_bridge(bridge)
anthropic_set_bridge(bridge)
search_set_bridge(bridge)
july_set_bridge(bridge)
calculator_set_bridge(bridge)
from app.events import event_manager
from fastapi.staticfiles import StaticFiles
import uuid
from app.context import request_id_var, acquired_instances_var

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the bridge which starts all orchestrators
    event_manager.start()
    await bridge.start()
    yield
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
async def http_metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start

    route = request.scope.get("route")
    path = route.path if route else request.url.path

    from july_telemetry.metrics import http_requests_total, http_request_duration_seconds
    http_requests_total.labels(
        method=request.method, path=path, status_code=str(response.status_code)
    ).inc()
    http_request_duration_seconds.labels(
        method=request.method, path=path
    ).observe(duration)

    return response

@app.middleware("http")
async def otel_tracing_middleware(request: Request, call_next):
    from opentelemetry import trace
    from opentelemetry.propagate import extract
    from opentelemetry.trace import SpanKind, Status, StatusCode

    carrier = {k.lower(): v for k, v in request.headers.items()}
    ctx = extract(carrier)

    correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    tracer = trace.get_tracer("july-engine")

    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}",
        context=ctx,
        kind=SpanKind.SERVER,
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.target": request.url.path,
            "correlation.id": correlation_id,
        },
    ) as span:
        try:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 500:
                span.set_status(Status(StatusCode.ERROR))
            response.headers["x-correlation-id"] = correlation_id
            return response
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Reuse correlation ID from gateway as the request ID when available
    rid = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    token_rid = request_id_var.set(rid)
    # Inicializa o registro de instâncias para esta requisição
    token_instances = acquired_instances_var.set({})
    _is_streaming = False

    try:
        from fastapi.responses import StreamingResponse

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
                _is_streaming = True
                original_iterator = response.body_iterator
                # Capture tokens so wrapped_iterator can reset them after the stream
                _token_rid = token_rid
                _token_instances = token_instances

                async def wrapped_iterator():
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    finally:
                        # Force-release any sequence slots acquired during this request
                        cleanup_explicit(current_acquired)
                        # Reset context vars only now — keeping request_id_var alive
                        # throughout the stream ensures SequencePool.release() sees
                        # the request ID and does NOT free the slot between agentic
                        # loop turns (tool call → second LLM turn).
                        request_id_var.reset(_token_rid)
                        acquired_instances_var.reset(_token_instances)

                response.body_iterator = wrapped_iterator()
                return response

            # Resposta síncrona: limpa aqui mesmo
            cleanup_explicit(current_acquired)
            return response
        finally:
            if 'current_acquired' not in locals():
                cleanup_explicit(acquired_instances_var.get())
    finally:
        # For streaming responses the cleanup/reset happens inside wrapped_iterator.
        # Only reset here for non-streaming (or early-error) paths.
        if not _is_streaming:
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
app.include_router(webhooks_router)
app.include_router(july_router)


@app.get("/health", tags=["July"])
async def health():
    return {"status": "online", "engine": "July Engine"}


@app.get("/metrics", tags=["July"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint — usado pelo Grafana stack."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
