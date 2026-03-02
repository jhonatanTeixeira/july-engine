#!/bin/bash
# local_models/setup.sh

# Exit on error
set -e

echo "-------------------------------------------------------"
echo "Configurando ambiente para RTX 3050 4GB (CUDA)..."
echo "-------------------------------------------------------"

# Variáveis para compilação do llama-cpp com suporte a GPU
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1

echo "[1/3] Instalando demais dependências do requirements.txt..."
pip install -r requirements.txt --no-cache-dir

echo "[2/3] Criando diretório de modelos..."
mkdir -p models

echo "[3/3] Baixando modelos GGUF (Aguarde, arquivos grandes)..."

# Dolphin 3.0 (Texto - Llama 3.1 8B Q4_K_M)
if [ ! -f "models/dolphin-8b.gguf" ]; then
    echo "Baixando Dolphin 3.0 GGUF..."
    curl -L "https://huggingface.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf?download=true" -o models/dolphin-8b.gguf
else
    echo "Dolphin 3.0 já existe, pulando download."
fi

# Llama 1b para testes (Texto - Llama 3.1 8B Q4_K_M)
if [ ! -f "models/qwen3-0.6b.gguf" ]; then
    echo "Baixando qwen3-0.6b para testes..."
    curl -L "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-IQ1_S.gguf?download=true" -o models/qwen3-0.6b.gguf
else
    echo "qwen3-0.6b já existe, pulando download."
fi

# BakLLaVA-1 (Visão - Q4_0)
if [ ! -f "models/bakllava-v1.gguf" ]; then
    echo "Baixando BakLLaVA-1 GGUF..."
    curl -L "https://huggingface.co/abetlen/BakLLaVA-1-GGUF/resolve/main/bakllava-1.Q4_0.gguf?download=true" -o models/bakllava-v1.gguf
else
    echo "BakLLaVA-1 já existe, pulando download."
fi

# BakLLaVA mmproj (Projector para Visão)
if [ ! -f "models/bakllava-v1-mmproj.gguf" ]; then
    echo "Baixando BakLLaVA mmproj..."
    curl -L "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf?download=true" -o models/bakllava-v1-mmproj.gguf
else
    echo "BakLLaVA mmproj já existe, pulando download."
fi

# Nanonets for testing
if [ ! -f "models/nanonets.gguf" ]; then
    echo "Baixando Nanonets GGUF for testing..."
    curl -L "https://huggingface.co/mradermacher/Nanonets-OCR2-1.5B-exp-i1-GGUF/resolve/main/Nanonets-OCR2-1.5B-exp.i1-IQ1_S.gguf?download=true" -o models/nanonets.gguf
else
    echo "nanonet já existe, pulando download."
fi

# Nanonet mmproj (Projector para Visão)
if [ ! -f "models/nanonets-mmproj.gguf" ]; then
    echo "Baixando Nanonets mmproj..."
    curl -L "https://huggingface.co/mradermacher/Nanonets-OCR2-1.5B-exp-GGUF/resolve/main/Nanonets-OCR2-1.5B-exp.mmproj-Q8_0.gguf?download=true" -o models/nanonets-mmproj.gguf
else
    echo "nanonet mmproj já existe, pulando download."
fi

# Emotion Recognition (Mood Analysis)
if [ ! -f "models/emotion-ferplus-8.onnx" ]; then
    echo "Baixando Emotion Recognition ONNX..."
    curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/emotion_ferplus/model/emotion-ferplus-8.onnx" -o models/emotion-ferplus-8.onnx
else
    echo "Emotion model já existe, pulando download."
fi

# Default Piper Model
if [ ! -f "models/en_US-lessac-medium.onnx" ]; then
    echo "Baixando default Piper model..."
    curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -o models/en_US-lessac-medium.onnx
    curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -o models/en_US-lessac-medium.onnx.json
else
    echo "Piper default model já existe, pulando download."
fi

echo "-------------------------------------------------------"
echo "Setup concluído com sucesso!"
echo "Modelos prontos em local_models/models/"
echo "-------------------------------------------------------"