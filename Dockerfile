# Stage 1: Build stage (CUDA Support)
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

# Install Python and build dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy requirements and setup script
COPY requirements.txt .
COPY setup.sh .

# Run setup.sh - This handles CUDA compilation for llama-cpp and downloads models
# We ensure it's executable and fix CRLF issues
RUN chmod +x setup.sh && sed -i 's/\r$//' setup.sh && ./setup.sh

# Copy the rest of the application
COPY . .

# Build the application with PyInstaller
RUN pip install pyinstaller
RUN pyinstaller --onefile --name july_engine jully_engine/main.py

# Stage 2: Runtime stage
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Install runtime dependencies (OpenCV, etc. might need these)
RUN apt-get update && apt-get install -y \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /app/dist/july_engine .
# Copy models downloaded by setup.sh
COPY --from=builder /app/models ./models
# Copy config files
COPY config.json . 
COPY voices.json . 

# Create storage folders
RUN mkdir -p storage/voices storage/temp

EXPOSE 8000

# Run the binary
CMD ["./july_engine"]
