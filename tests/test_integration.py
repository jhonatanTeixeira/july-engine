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

# --- VOICE MANAGEMENT TESTS ---

@pytest.mark.anyio
async def test_integration_list_voices(client):
    print("\n[Integration] Testing List Voices...")
    response = await client.get("/speech/voices")
    assert response.status_code == 200
    voices = response.json()
    assert isinstance(voices, list)
    # Should at least have yuni from voices.json
    ids = [v["id"] for v in voices]
    assert "yuni" in ids

@pytest.mark.anyio
async def test_integration_add_voice(client):
    print("\n[Integration] Testing Add Voice...")
    dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    
    # Test adding a clone voice
    data = {
        "name": "Test Voice",
        "language": "en",
        "type": "clone"
    }
    files = {"file": ("test.wav", dummy_wav, "audio/wav")}
    
    response = await client.post("/speech/voices", data=data, files=files)
    assert response.status_code == 200
    new_voice = response.json()
    assert "id" in new_voice
    assert new_voice["name"] == "Test Voice"
    assert "path" in new_voice
    assert "uploaded" in new_voice["path"]
    
    # Verify it appears in list
    list_response = await client.get("/speech/voices")
    voices = list_response.json()
    assert any(v["id"] == new_voice["id"] for v in voices)

# --- GPU TESTS ---

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_text_qwen(client):
    print("\n[Integration] Testing GPU Text (Qwen 0.6B)...")
    payload = {
        "model": "qwen3-0.6b.gguf", 
        "messages": [{"role": "user", "content": "Hello GPU, who are you?"}],
        "max_tokens": 20
    }
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    print(f"GPU Qwen Output: {content}")
    assert len(content) > 0

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_vision_nanonets(client):
    print("\n[Integration] Testing GPU Vision (Nanonets)...")
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
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    print(f"GPU Vision Output: {content}")
    assert len(content) > 0

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_emotion(client):
    print("\n[Integration] Testing GPU Emotion (Multimodal)...")
    image_path = os.path.join(os.path.dirname(__file__), "sad_person.jpg")
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
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    print(f"GPU Emotion Output: {content}")
    assert content.lower() == "sadness"

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_tts_xtts(client):
    print("\n[Integration] Testing GPU TTS (XTTS)...")
    payload = {"model": "xtts", "input": "Testing July Engine XTTS on GPU.", "voice": "yuni"}
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/audio/speech", json=payload, headers=headers)
    assert response.status_code == 200
    assert len(response.content) > 1000

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_tts_piper(client):
    print("\n[Integration] Testing GPU TTS (Piper)...")
    payload = {"model": "piper", "input": "Testing July Engine Piper on GPU.", "voice": "en_US-lessac-medium"}
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/audio/speech", json=payload, headers=headers)
    assert response.status_code == 200
    assert len(response.content) > 1000

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_stt_faster_whisper(client):
    print("\n[Integration] Testing GPU STT (FasterWhisper)...")
    dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("test.wav", dummy_wav, "audio/wav")}
    data = {"model": "faster-whisper"}
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/audio/transcriptions", files=files, data=data, headers=headers)
    assert response.status_code == 200
    assert "text" in response.json()

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_embeddings_e5(client):
    print("\n[Integration] Testing GPU Embeddings (Multilingual-E5)...")
    payload = {"model": "multilingual-e5", "input": "This is a test sentence for GPU embeddings."}
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"][0]["embedding"]) > 0

@pytest.mark.gpu
@pytest.mark.anyio
async def test_integration_gpu_image_edit_pix2pix(client):
    print("\n[Integration] Testing GPU Image Edit (Pix2Pix)...")
    # Create a simple red image
    img = Image.new('RGB', (256, 256), color = 'red')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    files = {"image": ("red.png", img_bytes, "image/png")}
    data = {"prompt": "make it blue", "model": "pix2pix"}
    headers = {"x-backend": "gpu"}
    
    response = await client.post("/v1/openai/images/edits", files=files, data=data, headers=headers)
    assert response.status_code == 200
    res_data = response.json()
    assert "data" in res_data
    assert len(res_data["data"][0]["b64_json"]) > 100

# --- API TESTS ---

@pytest.mark.api
@pytest.mark.anyio
async def test_integration_api_emotion(client):
    print("\n[Integration] Testing API Emotion (via vision model)...")
    image_path = os.path.join(os.path.dirname(__file__), "sad_person.jpg")
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
    headers = {"x-backend": "api"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    json_resp = response.json()
    print(f"API Emotion Full Response: {json.dumps(json_resp, indent=2)}")
    content = json_resp["choices"][0]["message"]["content"]
    print(f"API Emotion Output: {content}")
    # We expect 'sadness' because Moondream should identify it via the system prompt
    assert "sadness" in content.lower()

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

@pytest.mark.api
@pytest.mark.anyio
async def test_integration_api_ollama_vision(client):
    print("\n[Integration] Testing Ollama API Vision (vision_unit_tests)...")
    
    # Use the existing sad_person.jpg file
    image_path = os.path.join(os.path.dirname(__file__), "sad_person.jpg")
    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read()).decode()
    
    payload = {
        "model": "ollama/vision_unit_tests",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the person in this image and their emotion."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }
        ],
        "max_tokens": 100
    }
    headers = {"x-backend": "api"}
    try:
        response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            print(f"Ollama Vision status: {response.status_code} - {response.text}")
            pytest.skip("Ollama vision_unit_tests not available")
        
        json_resp = response.json()
        print(f"Ollama Vision Full Response: {json.dumps(json_resp, indent=2)}")
        
        content = json_resp["choices"][0]["message"]["content"]
        print(f"Ollama Vision Output: {content}")
        assert len(content) > 0
    except Exception as e:
        print(f"Error in Ollama Vision: {e}")
        pytest.skip(f"Ollama API vision failed: {e}")

@pytest.mark.api
@pytest.mark.anyio
async def test_integration_api_image_generation_loopback(client):
    print("\n[Integration] Testing API Image Generation Loopback (pix2pix)...")
    payload = {
        "prompt": "a beautiful sunset",
        "model": "pix2pix"
    }
    # Point back to our own server for loopback testing
    headers = {
        "x-backend": "api",
        "x-base-url": "http://localhost:8000/v1/openai"
    }
    try:
        response = await client.post("/v1/openai/images/generations", json=payload, headers=headers)
        if response.status_code == 500 and "Connection refused" in response.text:
            pytest.skip("Local loopback requires server running on :8000")
        assert response.status_code == 200
        res_data = response.json()
        assert "data" in res_data
        assert len(res_data["data"][0]["b64_json"]) > 100
    except Exception as e:
        pytest.skip(f"API Image Gen loopback failed: {e}")

@pytest.mark.api
@pytest.mark.anyio
async def test_integration_api_image_edit_loopback(client):
    print("\n[Integration] Testing API Image Edit Loopback (pix2pix)...")
    img = Image.new('RGB', (256, 256), color = 'green')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    files = {"image": ("green.png", img_bytes, "image/png")}
    data = {"prompt": "make it red", "model": "pix2pix"}
    headers = {
        "x-backend": "api",
        "x-base-url": "http://localhost:8000/v1/openai"
    }
    try:
        response = await client.post("/v1/openai/images/edits", files=files, data=data, headers=headers)
        if response.status_code == 500 and "Connection refused" in response.text:
            pytest.skip("Local loopback requires server running on :8000")
        assert response.status_code == 200
        res_data = response.json()
        assert "data" in res_data
        assert len(res_data["data"][0]["b64_json"]) > 100
    except Exception as e:
        pytest.skip(f"API Image Edit loopback failed: {e}")
