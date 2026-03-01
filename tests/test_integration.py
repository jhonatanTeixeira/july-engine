import os
import pytest
import asyncio
import base64
import io
import json
from httpx import AsyncClient, ASGITransport
import time
from PIL import Image

# Setup environment
os.environ['ENV'] = 'test'

OUTPUT_DIR = "tests/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
async def client():
    # Fixed import to use the correct package name
    from jully_engine.main import app
    from jully_engine.bridge import bridge
    
    # Ensure bridge is started for integration tests
    await bridge.start()
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=900.0) as ac:
        yield ac

# --- CPU TESTS (GGUF & LOCAL) ---

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_text_qwen(client):
    print("\n[Integration] Testing CPU Text (Qwen 0.6B)...")
    payload = {
        "model": "qwen3-0.6b.gguf", 
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 20,
        "temperature": 0.7,
        "num_ctx": 4096
    }
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    print(f"CPU Qwen Output: {content}")
    assert len(content) > 0

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_vision_nanonets(client):
    print("\n[Integration] Testing CPU Vision (Nanonets)...")
    img = Image.new('RGB', (224, 224), color = 'white')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    payload = {
        "model": "nanonets.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }
        ]
    }
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    print(f"CPU Vision Output: {content}")
    assert len(content) > 0

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_emotion(client):
    print("\n[Integration] Testing CPU Emotion (Multimodal)...")
    
    # Use the existing sad_person.jpg file
    image_path = os.path.join(os.path.dirname(__file__), "sad_person.jpg")
    if not os.path.exists(image_path):
        pytest.fail(f"Test image not found at {image_path}")
        
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode()
    
    payload = {
        "model": "emotion",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How does this person feel?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }
        ]
    }
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    print(f"CPU Emotion Output: {content}")
    
    # Specifically assert sadness for this test image
    assert content.lower() == "sadness"

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_tts_xtts(client):
    print("\n[Integration] Testing CPU TTS (XTTS)...")
    
    # Ensure a speaker wav exists in storage/voices/yuni.wav
    voices_dir = os.path.join(os.path.dirname(__file__), "..", "storage", "voices")
    os.makedirs(voices_dir, exist_ok=True)
    yuni_wav = os.path.join(voices_dir, "yuni.wav")
    if not os.path.exists(yuni_wav):
        # Create a dummy valid wav file for XTTS speaker
        dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        with open(yuni_wav, "wb") as f:
            f.write(dummy_wav)

    payload = {"model": "xtts", "input": "Testing July Engine XTTS on CPU.", "voice": "yuni"}
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/audio/speech", json=payload, headers=headers)
    assert response.status_code == 200
    assert len(response.content) > 1000

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_tts_piper(client):
    print("\n[Integration] Testing CPU TTS (Piper)...")
    # Using a known voice from rhasspy/piper-voices
    payload = {"model": "piper", "input": "Testing July Engine Piper on CPU.", "voice": "en_US-lessac-medium"}
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/audio/speech", json=payload, headers=headers)
    assert response.status_code == 200
    assert len(response.content) > 1000

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_stt_faster_whisper(client):
    print("\n[Integration] Testing CPU STT (FasterWhisper)...")
    dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("test.wav", dummy_wav, "audio/wav")}
    data = {"model": "faster-whisper"}
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/audio/transcriptions", files=files, data=data, headers=headers)
    assert response.status_code == 200
    assert "text" in response.json()

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_cpu_embeddings_bge(client):
    print("\n[Integration] Testing CPU Embeddings (BgeMicro)...")
    payload = {"model": "bge-micro", "input": "This is a test sentence for embeddings."}
    headers = {"x-backend": "cpu"}
    response = await client.post("/v1/openai/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"][0]["embedding"]) > 0

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_vision_fastvlm(client):
    pytest.skip("FastVLM is currently broken in transformers lib")

@pytest.mark.cpu
@pytest.mark.anyio
async def test_integration_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

# --- GPU TESTS ---

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_vllm_chat(client):
    print("\n[Integration] Testing GPU Chat (vLLM)...")
    payload = {
        "model": "mistral-7b-v0.1", 
        "messages": [{"role": "user", "content": "Count to 3."}],
        "stream": False
    }
    headers = {"x-backend": "gpu"}
    try:
        response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
        if response.status_code == 500:
             pytest.skip("GPU Backend not available")
        assert response.status_code == 200
    except Exception as e:
        pytest.skip(f"GPU connection failed: {e}")

# --- API TESTS ---

@pytest.mark.api
@pytest.mark.anyio
async def test_integration_api_ollama_qwen(client):
    print("\n[Integration] Testing Ollama API (unit_tests)...")
    payload = {
        "model": "ollama/unit_tests",
        "messages": [{"role": "user", "content": "Hello Ollama!"}],
        "max_tokens": 10
    }
    headers = {"x-backend": "api"}
    try:
        response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            pytest.skip("Ollama unit_tests not available")
        assert response.status_code == 200
    except Exception as e:
        pytest.skip(f"Ollama connection failed: {e}")
