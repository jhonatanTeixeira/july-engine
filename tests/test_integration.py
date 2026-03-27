import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pytest
import asyncio
import base64
import io
import json
from httpx import AsyncClient, ASGITransport
from PIL import Image

# Force the environment to test
os.environ['ENV'] = 'test'
os.environ['PERSISTENCE_BACKEND'] = 'tinydb'

@pytest.fixture(scope="module")
def anyio_backend():
    return 'asyncio'

@pytest.fixture(scope="module")
async def client():
    # Import inside fixture to ensure env vars are set
    from jully_engine.main import app
    from jully_engine.bridge import bridge
    from jully_engine.persistence.tinydb_backend import TinyDBBackend
    from jully_engine.persistence import persistence
    
    # Use a specific test database named 'testes.json'
    test_db_path = os.path.join("storage", "db", "testes.json")
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    backend = TinyDBBackend(test_db_path)
    persistence._backend_instance = backend
    
    # --- Inject Settings ---
    backend.set_setting("STT", {"model": "faster-whisper", "backend": "gpu"})
    backend.set_setting("TTS", {"model": "kokoro", "backend": "gpu", "voice": "af_sky", "language": "a"})
    backend.set_setting("VISION", {"model": "fastvlm", "backend": "gpu"})
    backend.set_setting("IMAGE_CREATE", {"model": "lcm", "backend": "gpu"})
    backend.set_setting("WEB_SEARCH", {"model": "tavily", "backend": "api"})
    
    text_presets = [
        {"alias": "qwen3-cpu", "model": "qwen3-0.6b", "backend": "cpu"},
        {"alias": "qwen3-gpu", "model": "qwen3-0.6b", "backend": "gpu"},
        {"alias": "qwen3-gpu-reasoning", "model": "qwen3-0.6b-reasoning", "backend": "gpu"},
        {"alias": "qwen3-gpu-mcp", "model": "qwen3-0.6b", "backend": "gpu", "mcp_option": "internal"}
    ]

    backend.set_setting("TEXT_PRESETS", text_presets)
    
    # --- Inject Models ---
    backend.set_model("qwen3-0.6b", {
            "model_alias": "Qwen3-0.6B-FP16",
            "model_type": "text",
            "model_id": "appleyu/Qwen3-0.6B-FP16-gguf",
            "filename": "Qwen3-0.6B-FP16.gguf",
            "mmproj_id": None,
            "mmproj_filename": None,
            "template": "chatml-function-calling",
            "context_window": 4096,
            "num_params": 0.6,
            "quantization": "F16",
            "num_layers": -1,
            "force_reasoning": None,
            "file_path": "C:\\Users\\jhona/.cache/huggingface/hub\\models--appleyu--Qwen3-0.6B-FP16-gguf\\snapshots\\421187a1573b0ac2be5466d7b45da087c5ee3367\\Qwen3-0.6B-FP16.gguf",
            "mmproj_path": None
    })
    
    backend.set_model("qwen3-0.6b-reasoning", {
            "model_alias": "Qwen3-0.6B-FP16-Reasoning",
            "model_type": "text",
            "model_id": "appleyu/Qwen3-0.6B-FP16-gguf",
            "filename": "Qwen3-0.6B-FP16.gguf",
            "mmproj_id": None,
            "mmproj_filename": None,
            "template": "chatml-function-calling",
            "context_window": 4096,
            "num_params": 0.6,
            "quantization": "F16",
            "num_layers": -1,
            "force_reasoning": True,
            "file_path": "C:\\Users\\jhona/.cache/huggingface/hub\\models--appleyu--Qwen3-0.6B-FP16-gguf\\snapshots\\421187a1573b0ac2be5466d7b45da087c5ee3367\\Qwen3-0.6B-FP16.gguf",
            "mmproj_path": None
    })
        
    await bridge.start()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=900.0) as ac:
        yield ac
    await bridge.stop()

# --- STT and TTS ---
@pytest.mark.anyio
async def test_stt_tts_integration(client):
    print("\n[Test] Running STT and TTS Integration...")
    tts_payload = {"model": "kokoro", "input": "The quick brown fox jumps over the lazy dog.", "voice": "af_sky"}
    headers = {"x-backend": "gpu"}
    tts_response = await client.post("/v1/openai/audio/speech", json=tts_payload, headers=headers)
    assert tts_response.status_code == 200
    audio_bytes = tts_response.content
    
    files = {"file": ("speech.wav", audio_bytes, "audio/wav")}
    stt_data = {"model": "faster-whisper"}
    stt_response = await client.post("/v1/openai/audio/transcriptions", files=files, data=stt_data, headers=headers)
    assert stt_response.status_code == 200
    transcribed_text = stt_response.json()["text"]
    print(f"Transcribed: {transcribed_text}")
    assert len(transcribed_text) > 0

# --- Image Generation and Vision ---
@pytest.mark.anyio
async def test_image_and_vision_integration(client):
    print("\n[Test] Running Image Generation and Vision Integration...")
    gen_payload = {
        "model": "lcm",
        "prompt": "generate an image of a womam with dark hair and glasses",
        "response_format": "b64_json"
    }
    headers_gpu = {"x-backend": "gpu"}
    gen_response = await client.post("/v1/openai/images/generations", json=gen_payload, headers=headers_gpu)
    assert gen_response.status_code == 200
    img_b64 = gen_response.json()["data"][0]["b64_json"]
    
    vision_payload = {
        "model": "qwen3-gpu",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "is this a woman with dark hair and glasses, answer just yes or no"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ]
    }
    headers = {}
    vision_response = await client.post("/v1/openai/chat/completions", json=vision_payload, headers=headers)
    assert vision_response.status_code == 200
    answer = vision_response.json()["choices"][0]["message"]["content"].strip().lower()
    print(f"Vision answer: {answer}")
    assert "yes" in answer or "no" in answer

# --- Chat OpenAI ---
@pytest.mark.anyio
async def test_chat_openai_integration(client):
    print("\n[Test] Running Chat OpenAI Integration...")
    payload = {
        "model": "qwen3-gpu",
        "messages": [
            {"role": "system", "content": "sempre que o usuario te perguntar seu nome, responda exatamante \"eu sou o batman\""},
            {"role": "user", "content": "qual o seu nome"}
        ]
    }
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/openai/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"].strip()
    print(f"Chat Output: {content}")
    assert "batman" in content.lower()

# --- Chat Anthropic ---
@pytest.mark.anyio
async def test_chat_anthropic_integration(client):
    print("\n[Test] Running Chat Anthropic Integration...")
    payload = {
        "model": "qwen3-gpu",
        "system": "sempre que o usuario te perguntar seu nome, responda exatamante \"eu sou o batman\"",
        "messages": [
            {"role": "user", "content": "qual o seu nome"}
        ],
        "max_tokens": 100
    }
    headers = {"x-backend": "gpu"}
    response = await client.post("/v1/anthropic/messages", json=payload, headers=headers)
    assert response.status_code == 200
    
    res_json = response.json()
    content = res_json["content"][0].get("text", "")
    reasoning = res_json.get("reasoning_content", "")
    print(f"Anthropic Output (Text): {content}")
    print(f"Anthropic Output (Reasoning): {reasoning}")
    assert "batman" in content.lower() or "batman" in reasoning.lower()

# --- Other Endpoints ---
@pytest.mark.anyio
async def test_other_endpoints(client):
    print("\n[Test] Running Other Endpoints Tests...")
    resp = await client.get("/models/gguf/")
    assert resp.status_code == 200
    
    resp = await client.get("/system/monitoring/")
    assert resp.status_code == 200
    
    calc_payload = {"num_params": 7, "quantization": "Q4_K_M", "context_window": 4096, "num_layers": 32}
    resp = await client.post("/system/check-resources", json=calc_payload)
    assert resp.status_code == 200
    
    search_payload = {"query": "test search"}
    resp = await client.post("/search/web", json=search_payload)
    assert resp.status_code in [200, 500, 501]

# --- OpenAI Streaming with Reasoning ---
@pytest.mark.anyio
async def test_openai_streaming_reasoning(client):
    print("\n[Test] Running OpenAI Streaming Reasoning Integration...")
    payload = {
        "model": "qwen3-gpu-reasoning",
        "messages": [
            {"role": "user", "content": "olá, tudo bem?"}
        ],
        "stream": True
    }
    headers = {"x-backend": "gpu"}
    
    reasoning_found = False
    content_found = False
    
    async with client.stream("POST", "/v1/openai/chat/completions", json=payload, headers=headers) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                
                if "reasoning_content" in delta:
                    reasoning_found = True
                    print(f"R: {delta['reasoning_content']}", end="", flush=True)
                
                if "content" in delta:
                    content_found = True
                    print(delta["content"], end="", flush=True)
    
    print("\n")
    assert reasoning_found, "Reasoning content was not streamed"
    assert content_found, "Main content was not streamed"

# --- Anthropic Streaming with Reasoning ---
@pytest.mark.anyio
async def test_anthropic_streaming_reasoning(client):
    print("\n[Test] Running Anthropic Streaming Reasoning Integration...")
    payload = {
        "model": "qwen3-gpu-reasoning",
        "messages": [
            {"role": "user", "content": "olá, tudo bem?"}
        ],
        "max_tokens": 100,
        "stream": True
    }
    headers = {"x-backend": "gpu"}
    
    # In Anthropic Bridge, we yield both reasoning and content as text_delta
    # in content_block_delta events.
    deltas_found = 0
    
    async with client.stream("POST", "/v1/anthropic/messages", json=payload, headers=headers) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:].strip()
                try:
                    chunk = json.loads(data_str)
                    if chunk.get("type") == "content_block_delta":
                        text = chunk.get("delta", {}).get("text", "")
                        if text:
                            deltas_found += 1
                            print(text, end="", flush=True)
                except json.JSONDecodeError:
                    continue
    
    print("\n")
    assert deltas_found > 0, "No content deltas were streamed in Anthropic mode"

# --- Internal MCP Image Generation (Non-Stream and Stream) ---
@pytest.mark.anyio
async def test_internal_mcp_image_generation(client):
    print("\n[Test] Running Internal MCP Image Generation Integration...")
    
    # 1. Non-Stream Mode
    print("Testing Non-Stream MCP...")
    payload_sync = {
        "model": "qwen3-gpu-mcp",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with image generation capabilities. When asked to generate an image, use the <generate_image><prompt>...</prompt></generate_image> tool."},
            {"role": "user", "content": "Gere uma imagem de um gato de óculos"}
        ],
        "stream": False
    }
    headers = {
        "x-backend": "gpu",
        "x-enable-internal-mcp": "1"
    }
    
    # Force the model to use the tool in its response (it's internal tool calling)
    # The internal MCP will inject the tool definition.
    
    response = await client.post("/v1/openai/chat/completions", json=payload_sync, headers=headers)
    assert response.status_code == 200
    res_json = response.json()
    
    # The InternalMCP returns the image as a b64 string in the content if successful
    content = res_json["choices"][0]["message"]["content"]
    
    # Verify if it contains a base64 image (or is a list containing image object)
    image_found = False
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part):
                image_found = True
                break
    elif isinstance(content, str):
        if "data:image/png;base64," in content or "data:image/jpeg;base64," in content:
            image_found = True
        
    assert image_found, f"Image was not generated in non-stream MCP mode. Content received: {content}"
    print("Non-Stream MCP Image Generation: OK")

    # 2. Stream Mode
    print("Testing Stream MCP...")
    payload_stream = {
        "model": "qwen3-gpu-mcp",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with image generation capabilities. When asked to generate an image, use the <generate_image><prompt>...</prompt></generate_image> tool."},
            {"role": "user", "content": "Gere uma imagem de um gato de óculos"}
        ],
        "stream": True
    }
    
    stream_image_found = False
    async with client.stream("POST", "/v1/openai/chat/completions", json=payload_stream, headers=headers) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]": break
                
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                
                if delta.get("type") == "image_url" or "image_url" in delta:
                    stream_image_found = True
                    print("[Image Chunk Found]", flush=True)
    
    assert stream_image_found, "Image was not generated in stream MCP mode"
    print("Stream MCP Image Generation: OK")
