#!/bin/bash

# Exit on error
set -e

echo "-------------------------------------------------------"
echo "Configurando ambiente para RTX 3050 4GB (CUDA)..."
echo "-------------------------------------------------------"

WHEELS_DIR="./vendor/llama-cpp-python/dist"
SUBMODULE_DIR="./vendor/llama-cpp-python"
CUDA_ARCH="86"  # RTX 3050 = sm_86

# Detecta a versão do Python no venv (ex: 311, 312)
PY_VER=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

# Detecta a wheel já compilada para a versão correta
EXISTING_WHEEL=$(find "$WHEELS_DIR" -name "llama_cpp_python-*-cp${PY_VER}-*.whl" 2>/dev/null | head -n 1)

# Suporte para forçar recompilação via RECOMPILE=true
if [ "$RECOMPILE" = "true" ] || [ -z "$EXISTING_WHEEL" ]; then
    echo "[1/3] Forçando recompilação do llama-cpp-python..."
    # Limpa builds anteriores para garantir nova compilação
    rm -rf "$WHEELS_DIR"
    rm -rf "$SUBMODULE_DIR/build"

    echo "[1/3] Compilando llama-cpp-python com CUDA (apenas desta vez)..."

    # git submodule update --init --recursive

    if [ ! -d "$SUBMODULE_DIR" ]; then
        echo "ERRO: Submódulo não encontrado em $SUBMODULE_DIR"
        exit 1
    fi

    mkdir -p "$WHEELS_DIR"

    # GGML_SCHED_MAX_SPLIT_INPUTS=512 resolve crashes em modelos MoE/DeepSeek com muitos experts
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DGGML_SCHED_MAX_SPLIT_INPUTS=512" \
    CFLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=512" \
    CXXFLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=512" \
        .venv/bin/pip wheel "$SUBMODULE_DIR" \
        --no-deps \
        -w "$WHEELS_DIR"

    EXISTING_WHEEL=$(find "$WHEELS_DIR" -name "llama_cpp_python-*-cp${PY_VER}-*.whl" | head -n 1)
    echo "      Wheel gerada: $EXISTING_WHEEL"
else
    echo "[1/3] Wheel do llama-cpp-python já existe: $EXISTING_WHEEL"
    echo "      Pulando compilação (use RECOMPILE=true para forçar)."
fi

echo "[2/3] Instalando dependencias base (requirements.txt)..."
.venv/bin/pip install -r requirements.txt # --no-cache-dir

echo "-------------------------------------------------------"
echo "Setup concluído com sucesso!"
echo "-------------------------------------------------------"