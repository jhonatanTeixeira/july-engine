#!/bin/bash

# Exit on error
set -e

echo "-------------------------------------------------------"
echo "Configurando ambiente para RTX 3050 4GB (CUDA)..."
echo "-------------------------------------------------------"

WHEELS_DIR="./vendor/llama-cpp-python/dist"
SUBMODULE_DIR="./vendor/llama-cpp-python"
CUDA_ARCH="86"  # RTX 3050 = sm_86

# Detecta a wheel já compilada para Python 3.12
EXISTING_WHEEL=$(find "$WHEELS_DIR" -name "llama_cpp_python-*-cp312-*.whl" 2>/dev/null | head -n 1)

if [ -n "$EXISTING_WHEEL" ]; then
    echo "[1/3] Wheel do llama-cpp-python já existe: $EXISTING_WHEEL"
    echo "      Pulando compilação."
else
    echo "[1/3] Compilando llama-cpp-python com CUDA (apenas desta vez)..."
    git submodule update --init --recursive

    if [ ! -d "$SUBMODULE_DIR" ]; then
        echo "ERRO: Submódulo não encontrado em $SUBMODULE_DIR"
        exit 1
    fi

    mkdir -p "$WHEELS_DIR"

    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
        .venv/bin/pip wheel "$SUBMODULE_DIR" \
        --no-deps \
        --no-cache-dir \
        -w "$WHEELS_DIR"

    EXISTING_WHEEL=$(find "$WHEELS_DIR" -name "llama_cpp_python-*-cp312-*.whl" | head -n 1)
    echo "      Wheel gerada: $EXISTING_WHEEL"
fi

echo "[2/3] Instalando dependencias base (requirements.txt)..."
.venv/bin/pip install -r requirements.txt --no-cache-dir

if [ "$ENABLE_CPU" = "true" ]; then
    echo "[2.1/3] Instalando dependencias para CPU (requirements_cpu.txt)..."
    .venv/bin/pip install -r requirements_cpu.txt --no-cache-dir
fi

if [ "$ENABLE_GPU" = "true" ]; then
    echo "[2.2/3] Instalando dependencias para GPU (requirements_gpu.txt)..."
    .venv/bin/pip install -r requirements_gpu.txt --no-cache-dir
fi

echo "-------------------------------------------------------"
echo "Setup concluído com sucesso!"
echo "-------------------------------------------------------"