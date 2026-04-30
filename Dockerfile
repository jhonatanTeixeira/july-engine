# Stage 1: Build & Runtime Environment (CUDA Support)
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Configurações de ambiente para Python e CUDA
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalação do Python 3.11 e dependências de compilação
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    python3.11-venv \
    git \
    cmake \
    curl \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Configura Python 3.11 como padrão
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instalação via setup.sh para garantir compilação do llama-cpp com flags de ambiente
# O setup.sh cria e configura o .venv automaticamente
RUN python -m venv .venv && \
    chmod +x setup.sh && \
    sed -i 's/\r$//' setup.sh && \
    ./setup.sh

# Criação de pastas de armazenamento
RUN mkdir -p storage/voices storage/temp

# Exposição da porta do FastAPI
EXPOSE 8000

# Healthcheck interno do Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Inicialização usando o uvicorn dentro do venv configurado pelo setup.sh
CMD [".venv/bin/python", "-m", "uvicorn", "jully_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
