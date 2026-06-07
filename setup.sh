#!/bin/bash

# Exit on error
set -e

# Install Python 3.11 and llama.cpp compilation requirements
echo "Checking and installing dependencies for july_engine..."
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y build-essential cmake make gcc g++ python3.11 python3.11-venv python3.11-dev


if [ "$SETUP_UBUNTU" = "true" ]; then
    echo "Instalando dependências de OCR do Tesseract..."
    sudo apt install -y tesseract-ocr tesseract-ocr-por
fi

# Verifica qual ambiente estamos configurando para exibir a mensagem correta
echo "-------------------------------------------------------"
if [ "$WITH_VULKAN" = "true" ]; then
    echo "Configurando ambiente para VULKAN (Placas AMD/Intel)..."
else
    echo "Configurando ambiente para RTX 3050 4GB (CUDA)..."
fi
echo "-------------------------------------------------------"

WHEELS_DIR="../july_engine_libs/python/llama_gguf/vendor/llama-cpp-python/dist"
SUBMODULE_DIR="../july_engine_libs/python/llama_gguf/vendor/llama-cpp-python"
# Define CUDA_ARCH com o valor da variável de ambiente, 
# ou usa "86" (RTX 3050) como padrão se estiver vazia.
CUDA_ARCH="${CUDA_ARCH:-86}"

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

    # git submodule update --init --recursive

    if [ ! -d "$SUBMODULE_DIR" ]; then
        echo "ERRO: Submódulo não encontrado em $SUBMODULE_DIR"
        exit 1
    fi

    mkdir -p "$WHEELS_DIR"

    # --- INÍCIO DA MODIFICAÇÃO CIRÚRGICA ---
    # Define os argumentos de compilação dependendo da variável de ambiente
    if [ "$WITH_VULKAN" = "true" ]; then
        echo "      -> Aplicando flags de compilação para VULKAN..."
        # Removemos os comandos CUDA e inserimos o -DGGML_VULKAN=on
        LOCAL_CMAKE_ARGS="-DGGML_VULKAN=on -DGGML_SCHED_MAX_SPLIT_INPUTS=512"
    else
        echo "      -> Aplicando flags de compilação para CUDA..."
        # Mantém a configuração original para a RTX 3050
        LOCAL_CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DGGML_SCHED_MAX_SPLIT_INPUTS=2048"
    fi
    # --- FIM DA MODIFICAÇÃO CIRÚRGICA ---

    # GGML_SCHED_MAX_SPLIT_INPUTS=512 resolve crashes em modelos MoE/DeepSeek com muitos experts
    # Passamos a nossa variável LOCAL_CMAKE_ARGS dinamicamente
    CMAKE_ARGS="$LOCAL_CMAKE_ARGS" \
    CFLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=2048" \
    CXXFLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=2048" \
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