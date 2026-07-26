# July Engine — Motor de Inferência 100% Local

O `july_engine` é o núcleo de inteligência do ecossistema Jully. Um motor de inferência multimodal construído com Python e FastAPI, capaz de processar texto, visão, áudio e imagem usando modelos GGUF locais — com gerenciamento automático e inteligente de VRAM/RAM.

**Nota:** A partir desta versão, July Engine é dedicado exclusivamente a inferência local. Serviços de busca externa (web search, code search) foram movidos para o serviço **July Search**, mantido em repositório próprio.

📖 **[Documentação completa](https://jhonatanteixeira.github.io/july-engine/)** — arquitetura, cada modelo local (um por página), calculadora de VRAM/RAM, painel administrativo e referência de API.

## O que torna o July Engine excepcional

### Gerenciamento Automático de VRAM/RAM
Nenhuma configuração manual de "quantos modelos cabem na GPU". O engine:
1. Calcula VRAM necessária antes de carregar cada modelo
2. Descarga o modelo menos recentemente usado (LRU) quando necessário
3. Reduz layers de GPU gradualmente se ainda não cabe
4. Mantém modelos"quentes" entre requests para zero cold-start

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
- **PyTorch + Transformers**: VLMs, Whisper, embeddings locais
- **Diffusers**: Stable Diffusion, Flux, AnimateDiff
- **DeepFace + OpenCV**: Reconhecimento facial local
- **ChromaDB + pgvector**: Busca vetorial e RAG local
- **MCP (internal)**: Model Context Protocol para tool calling interno

## Padrões de Projeto

- **Bridge Pattern**: `bridge.py` roteia sem conhecimento de domínio
- **Adapter Pattern**: `app/adapters/` — interface unificada por tipo de tarefa
- **Orchestrator Pattern**: `orchestrator.py` — gestão de recursos e concorrência
- **Service Locator**: `model_loader.py` — lazy loading com cache thread-safe

## API Endpoints

### Chat Textual
```
POST /v1/openai/chat/completions   # OpenAI-compatible
POST /v1/anthropic/messages        # Anthropic-compatible
```

### Áudio (TTS e STT Locais)
```
POST /v1/openai/audio/speech           # TTS streaming lokal
POST /v1/anthropic/audio/transcriptions # STT local
```

### Visão (Modelos Locais)
```
POST /july/v1/vision/images/describe   # Image captioning via Moondream/FastVLM
POST /july/v1/vision/video/describe    # Video analysis
POST /july/v1/vision/faces/extract     # Face detection + embedding local
POST /july/v1/vision/face/sync         # Batch face matching
```

### RAG / Memória Local
```
POST /july/v1/rag/add          # Adiciona memória ao vector store local
POST /july/v1/rag/search       # Busca semântica via embeddings locais
POST /july/v1/rag/smart-search # Busca + sumarização LLM local
POST /july/v1/rag/batch-add    # Inserção em lote
```

### Geração de Imagem (Local)
```
POST /v1/images/generations            # Flux/Stable Diffusion local
POST /july/v1/image-edit               # Pix2Pix instrucional
POST /july/v1/image-remove-background  # rembg
```

### Admin e Observabilidade
```
GET  /health                 # Status + VRAM/RAM disponível
GET  /api/models             # Modelos configurados
GET  /api/settings           # Configuração completa
POST /api/settings/update    # Hot-swap de modelos
```

## Headers de Controle

```http
x-backend: gpu          # GPU local (padrão)
x-backend: cpu          # CPU puro
x-context-window: 8192  # Janela de contexto específica
x-session-id: abc123    # ID de sessão para KV cache pooling
```

**Nota:** O header `x-backend: api` foi removido. Todas as requisições são processadas localmente.

## Configuração

Toda a configuração é dinâmica e persistida em banco (TinyDB por padrão, ou Postgres) — **não há** um `settings.yaml` estático apesar do que documentação antiga possa sugerir. Configure via:

- **[Admin Panel](https://jhonatanteixeira.github.io/july-engine/configuration/admin-panel/)** em `/admin` — recomendado, com estimativa de VRAM em tempo real antes de baixar um modelo.
- **API direta** — `GET`/`POST /v1/settings`, ou as rotas `/models/gguf/*` (`app/routers/models.py`) especificamente para o catálogo de modelos GGUF de chat.

Exemplo de shape de uma entrada em `TEXT_PRESETS` (não é um arquivo — é o JSON que fica armazenado no backend de persistência):

```json
{
  "alias": "qwen-7b",
  "model_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
  "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
  "backend": "gpu",
  "context_window": 8192,
  "flash_attn": true,
  "kv_cache_quantization": "Q8_0",
  "n_seq_max": 2,
  "is_default": true
}
```

Modelos MoE (Mixture-of-Experts) também suportam `cpu_moe`/`n_cpu_moe` para liberar VRAM mantendo os pesos de especialistas na CPU — veja [Resource Calculator](https://jhonatanteixeira.github.io/july-engine/configuration/resource-calculator/).

## Instalação

### Com CUDA (NVIDIA)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 3000
```

### Com Vulkan (AMD/Intel)
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 3000
```

## Relacionamento com July Search

Para funcionalidades que requerem buscas externas (web search via Tavily, code search via GitHub), utilize o serviço **July Search** complementarmente em `http://localhost:8001`. Os routers de busca são compartilhados entre ambos os serviços.

## Documentação Completa

A documentação oficial e completa do projeto — arquitetura (Bridge, Orchestrator, Model Loader, Adapters), cada modelo local documentado individualmente (~30 wrappers em `app/models/`, incluindo os que estão quebrados/dead code, documentados honestamente), a calculadora de VRAM/RAM (incluindo `vision_on_cpu` e as opções MoE `cpu_moe`/`n_cpu_moe`), o painel administrativo (`/admin`) e a referência completa de API — vive em **[jhonatanteixeira.github.io/july-engine](https://jhonatanteixeira.github.io/july-engine/)**, gerada a partir da pasta `docs/` deste repositório via [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

Para editar a documentação localmente:

```bash
pip install mkdocs mkdocs-material
mkdocs serve   # http://localhost:8000, com live-reload
```

O deploy para GitHub Pages é automático via `.github/workflows/docs.yml` a cada push em `docs/`/`mkdocs.yml` na branch `master`.

## Dependências

Os pacotes editáveis locais (`llama_gguf`, `telemetry`, `routers`) vêm do submódulo `vendor/july_engine_libs` ([july-engine-libs](https://github.com/jhonatanTeixeira/july-engine-libs)). Ao clonar, inicialize os submódulos antes de instalar:

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

## License

MIT — veja [LICENSE](LICENSE). Este projeto depende de bibliotecas de terceiros com licenças mais restritivas (incluindo AGPL-3.0 e GPL-3.0, além de pesos de modelos não-comerciais); veja [NOTICE.md](NOTICE.md) antes de qualquer uso comercial ou como serviço de rede.
