#!/bin/bash

# Exit on error
set -e

echo "-------------------------------------------------------"
echo "Configurando ambiente para RTX 3050 4GB (CUDA)..."
echo "-------------------------------------------------------"

export CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86"
# export FORCE_CMAKE=1

# echo "[1/3] Garantindo ferramentas de build no pip..."
# pip install --upgrade pip wheel setuptools

# echo "[2/3] Instalando llama-cpp-python com CUDA..."
# pip install llama-cpp-python --no-cache-dir

echo "[2/3] Instalando dependencias base (requirements.txt)..."
pip install -r requirements.txt --no-cache-dir

if [ "$ENABLE_CPU" = "true" ]; then
    echo "[2.1/3] Instalando dependencias para CPU (requirements_cpu.txt)..."
    pip install -r requirements_cpu.txt --no-cache-dir
fi

if [ "$ENABLE_GPU" = "true" ]; then
    echo "[2.2/3] Instalando dependencias para GPU (requirements_gpu.txt)..."
    pip install -r requirements_gpu.txt --no-cache-dir
fi

echo "[3/3] Criando diretório de modelos..."
mkdir -p models

if [ "$STARTUP_MODELS" = "none" ]; then
    echo "STARTUP_MODELS=none, pulando download de modelos."
else
    echo "Baixando modelos GGUF (Aguarde, arquivos grandes)..."

    # Dolphin 3.0 (Texto - Llama 3.1 8B Q4_K_M)
    if [ ! -f "models/dolphin-8b.gguf" ]; then
        echo "Baixando Dolphin 3.0 GGUF..."
        curl -L "https://huggingface.co/bartowski/Dolphin3.0-Llama3.1-8B-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf?download=true" -o models/dolphin-8b.gguf
    fi

    # Llama 1b para testes
    if [ ! -f "models/qwen3-0.6b.gguf" ]; then
        echo "Baixando qwen3-0.6b..."
        curl -L "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-IQ1_S.gguf?download=true" -o models/qwen3-0.6b.gguf
    fi

    # BakLLaVA-1
    if [ ! -f "models/bakllava-v1.gguf" ]; then
        echo "Baixando BakLLaVA-1 GGUF..."
        curl -L "https://huggingface.co/abetlen/BakLLaVA-1-GGUF/resolve/main/bakllava-1.Q4_0.gguf?download=true" -o models/bakllava-v1.gguf
    fi

    # BakLLaVA mmproj
    if [ ! -f "models/bakllava-v1-mmproj.gguf" ]; then
        echo "Baixando BakLLaVA mmproj..."
        curl -L "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf?download=true" -o models/bakllava-v1-mmproj.gguf
    fi

    # Nanonets
    if [ ! -f "models/nanonets.gguf" ]; then
        echo "Baixando Nanonets GGUF..."
        curl -L "https://huggingface.co/mradermacher/Nanonets-OCR2-1.5B-exp-i1-GGUF/resolve/main/Nanonets-OCR2-1.5B-exp.i1-IQ1_S.gguf?download=true" -o models/nanonets.gguf
    fi

    if [ ! -f "models/nanonets-mmproj.gguf" ]; then
        echo "Baixando Nanonets mmproj..."
        curl -L "https://huggingface.co/mradermacher/Nanonets-OCR2-1.5B-exp-GGUF/resolve/main/Nanonets-OCR2-1.5B-exp.mmproj-Q8_0.gguf?download=true" -o models/nanonets-mmproj.gguf
    fi

    # Emotion Recognition
    if [ ! -f "models/emotion-ferplus-8.onnx" ]; then
        echo "Baixando Emotion Recognition ONNX..."
        curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/emotion_ferplus/model/emotion-ferplus-8.onnx" -o models/emotion-ferplus-8.onnx
    fi

    # Default Piper Model
    if [ ! -f "models/en_US-lessac-medium.onnx" ]; then
        echo "Baixando default Piper model..."
        curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -o models/en_US-lessac-medium.onnx
        curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -o models/en_US-lessac-medium.onnx.json
    fi
fi

echo "-------------------------------------------------------"
echo "Setup concluído com sucesso!"
echo "-------------------------------------------------------"
