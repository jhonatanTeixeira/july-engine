# july_engine — Motor de Inferência Multimodal

O `july_engine` é o núcleo de inteligência do ecossistema Jully. Um motor de inferência multimodal construído com Python e FastAPI, capaz de processar texto, visão, áudio e imagem usando modelos GGUF locais ou APIs cloud — com gerenciamento automático e inteligente de VRAM.

## O que torna o july_engine excepcional

### Gerenciamento Automático de VRAM

Nenhuma configuração manual de "quantos modelos cabem na GPU". O engine:
1. Calcula VRAM necessária antes de carregar cada modelo
2. Descarga o modelo menos recentemente usado (LRU) quando necessário
3. Reduz layers de GPU gradualmente se ainda não cabe
4. Mantém modelos "quentes" entre requests para zero cold-start

### Múltiplas Requests Simultâneas por Modelo

Via **Sequence Pooling**: uma instância GGUF atende N requests paralelas sem duplicar VRAM:

```yaml
n_seq_max: 2  # 2 requests simultâneas, apenas +KV_Cache de overhead
```

### Compatibilidade Total com OpenAI API

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:3000/v1/openai", api_key="any")
# Funciona com qualquer modelo GGUF configurado
```

## Stack

- **FastAPI + uvicorn**: API assíncrona de alta performance
- **llama-cpp-python**: Inferência GGUF (CUDA, Vulkan, CPU)
- **PyTorch + Transformers**: VLMs, Whisper, embeddings
- **Diffusers**: Stable Diffusion, Flux, AnimateDiff
- **DeepFace + OpenCV**: Reconhecimento facial
- **ChromaDB + pgvector**: Busca vetorial
- **LiteLLM**: Dispatch para APIs cloud
- **MCP**: Model Context Protocol para tool calling

## Padrões de Projeto

- **Bridge Pattern**: `bridge.py` roteia sem conhecimento de domínio
- **Adapter Pattern**: `app/adapters/` — interface unificada por tipo de tarefa
- **Orchestrator Pattern**: `orchestrator.py` — gestão de recursos e concorrência
- **Service Locator**: `model_loader.py` — lazy loading com cache thread-safe

## API Endpoints

### Chat
```
POST /v1/openai/chat/completions   # OpenAI-compatible
POST /v1/anthropic/messages        # Anthropic-compatible
```

### Áudio
```
POST /v1/openai/audio/speech           # TTS streaming
POST /v1/anthropic/audio/transcriptions # STT
```

### Visão
```
POST /july/v1/vision/images/describe   # Image captioning
POST /july/v1/vision/video/describe    # Video analysis
POST /july/v1/vision/faces/extract     # Face detection + embedding
POST /july/v1/vision/face/sync         # Batch face matching
```

### RAG / Memória
```
POST /july/v1/rag/add          # Adiciona memória
POST /july/v1/rag/search       # Busca semântica
POST /july/v1/rag/smart-search # Busca + sumarização LLM
POST /july/v1/rag/batch-add    # Inserção em lote
```

### Imagens
```
POST /v1/images/generations            # Flux/Stable Diffusion
POST /july/v1/image-edit               # Pix2Pix instrucional
POST /july/v1/image-remove-background  # rembg
```

### Admin
```
GET  /health                 # Status + VRAM disponível
GET  /api/models             # Modelos configurados
GET  /api/settings           # Configuração completa
POST /api/settings/update    # Hot-swap de modelos
```

## Headers de Controle

```http
x-backend: gpu          # GPU local (padrão)
x-backend: cpu          # CPU puro
x-backend: api          # API cloud
x-context-window: 8192  # Janela de contexto específica
x-session-id: abc123    # ID de sessão para KV cache
```

## Configuração (settings.yaml)

```yaml
TEXT_PRESETS:
  - alias: "qwen-7b"
    model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF"
    filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    backend: "gpu"
    context_window: 8192
    flash_attn: true
    kv_cache_quantization: "Q8_0"
    n_seq_max: 2
    is_default: true

TTS:
  alias: "kokoro"
  model: "kokoro"
  backend: "cpu"

VISION:
  alias: "moondream"
  model_id: "vikhyatk/moondream2"
  model_type: "vision"
  backend: "gpu"
```

## Instalação

```bash
# Com CUDA (NVIDIA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -r requirements.txt

# Com Vulkan (AMD/Intel)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
pip install -r requirements.txt

# Iniciar
uvicorn main:app --host 0.0.0.0 --port 3000
```

## Documentação Completa

- [Arquitetura do Engine](../docs/docs/arquitetura/july_engine.md)
- [O Orchestrator](../docs/docs/arquitetura/orquestrador.md)
- [O Bridge](../docs/docs/arquitetura/bridge.md)
- [O Model Loader](../docs/docs/arquitetura/model_loader.md)
- [Biblioteca llama_gguf](../docs/docs/python/llama_gguf_lib.md)
- [Formato GGUF](../docs/docs/engine/gguf.md)
