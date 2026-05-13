import pytest
import json
import os
import base64
import time
import io
import asyncio
from httpx import AsyncClient, ASGITransport

os.environ["ENV"] = "test"

from main import app
from app.bridge import bridge
from app.persistence import get_backend

LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")
BASE_URL = "https://api.deepinfra.com/v1/openai"

@pytest.fixture
async def client():
    # Setup settings and presets
    backend = get_backend()
    
    # 1. Configurar Modelos em SETTINGS
    # Note: Using prefix 'openai/' for litellm compatibility where needed
    models_config = {
        "IMAGE_CREATE": {
            "model": "openai/black-forest-labs/FLUX-2-klein-4b",
            "base_url": BASE_URL,
            "api_key": LLM_API_TOKEN,
            "backend": "api"
        },
        "IMAGE_EDIT": {
            "model": "PrunaAI/p-image-Edit", # No prefix as litellm doesn't support it (manual call)
            "base_url": BASE_URL,
            "api_key": LLM_API_TOKEN,
            "backend": "api"
        },
        "EMBEDDINGS": {
            "model": "multilingual-e5",
            "backend": "cpu"
        },
        "TTS": {
            "model": "openai/hexgrad/Kokoro-82M",
            "base_url": BASE_URL,
            "api_key": LLM_API_TOKEN,
            "backend": "api"
        },
        "STT": {
            "model": "openai/openai/whisper-large-v3-turbo",
            "base_url": BASE_URL,
            "api_key": LLM_API_TOKEN,
            "backend": "api"
        }
    }
    
    for key, val in models_config.items():
        backend.set_setting(key.upper(), val)

    # 2. Configurar TEXT_PRESETS
    text_presets = [
        {
            "alias": "Euryale-70B",
            "model": "openai/Sao10K/L3.1-70B-Euryale-v2.2",
            "api_key": LLM_API_TOKEN,
            "backend": "api",
            "base_url": BASE_URL,
            "is_vision": False,
            "is_default": True,
            "mcp_option": "emulated"
        },
        {
            "alias": "Qwen-4B",
            "model": "openai/Qwen/Qwen3.5-4B",
            "api_key": LLM_API_TOKEN,
            "backend": "api",
            "base_url": BASE_URL,
            "is_vision": True,
            "is_default": False,
            "mcp_option": "internal"
        }
    ]
    backend.set_setting("TEXT_PRESETS", text_presets)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=900.0) as ac:
        yield ac
    await bridge.stop()

@pytest.mark.anyio
async def test_1_llm_image_orchestration(client):
    print("\n[Test 1] LLM Image Gen & Vision Verification...")
    
    # Step A: LLM Generates Image (Non-stream)
    # Using Euryale with emulated tools
    gen_prompt = "Gere uma imagem de uma garota feliz em um parque."
    payload = {
        "model": "Euryale-70B",
        "messages": [{"role": "user", "content": gen_prompt}],
        "stream": False
    }
    
    print("A1: Requesting image generation via LLM tool call...")
    response = await client.post("/v1/openai/chat/completions", json=payload, headers={"x-enable-internal-mcp": "1"})
    assert response.status_code == 200, response.text
    res_json = response.json()
    content = res_json["choices"][0]["message"]["content"]
    assert content, response.text
    print(f"LLM Response: {content[:100]}...")
    
    # Check if an image was actually "generated" (simulated by tool output in bridge/adapter)
    # Since we are testing API integration, we expect the tool call to trigger image_generation
    assert "image_url" in str(res_json) or "![image]" in content
    
    # For testing purposes, let's grab a real image generated if we can, 
    # but the prompt asks the LLM to do it. 
    # If the LLM just returns text, we need to ensure the tool was called.
    
    # Step B: Vision Verification (Stream)
    # Using Qwen-4B
    print("B1: Verifying image content via Vision (Stream)...")
    # We'll use a placeholder or the generated one if available
    # For the sake of a reliable integration test, we'll generate one directly first to have a B64
    gen_res = await client.post("/v1/openai/images/generations", json={
        "prompt": "A happy girl in a park, 8k",
        "model": "openai/black-forest-labs/FLUX-2-klein-4b"
    })
    assert gen_res.status_code == 200
    img_b64 = gen_res.json()["data"][0]["b64_json"]
    
    vision_payload = {
        "model": "Qwen-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Esta imagem contém uma garota feliz? Responda apenas sim ou não."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "stream": True
    }
    
    async with client.stream("POST", "/v1/openai/chat/completions", json=vision_payload) as resp:
        assert resp.status_code == 200
        full_text = ""
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                data = json.loads(line[6:])
                if data.get ("choices"):
                    full_text += data["choices"][0]["delta"].get("content", "") or ""
    
    print(f"Vision Answer: {full_text}")
    assert "sim" in full_text.lower() or "yes" in full_text.lower()

    # Step C: Image Edit
    print("C1: Editing image...")
    # Mocking UploadFile
    files = {
        "image": ("happy_girl.png", base64.b64decode(img_b64), "image/png")
    }
    data = {
        "prompt": "Transforme o dia em ensolarado e adicione flores ao redor da garota",
        "model": "PrunaAI/p-image-Edit"
    }
    edit_res = await client.post("/v1/openai/images/edits", files=files, data=data)
    assert edit_res.status_code == 200
    edited_b64 = edit_res.json()["data"][0]["b64_json"]
    print("Image edited successfully.")

    # Step D: Final Verification
    print("D1: Final verification of edited image...")
    vision_payload["messages"][1]["role"] = "user" # Ensure structure
    vision_payload["messages"][0]["content"][0]["text"] = "A imagem agora tem flores? Responda sim ou não."
    vision_payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{edited_b64}"
    vision_payload["stream"] = False
    
    final_res = await client.post("/v1/openai/chat/completions", json=vision_payload)
    assert final_res.status_code == 200
    final_text = final_res.json()["choices"][0]["message"]["content"]
    print(f"Final Vision Answer: {final_text}")
    assert "sim" in final_text.lower() or "yes" in final_text.lower()

@pytest.mark.anyio
async def test_2_stt_tts_api(client):
    print("\n[Test 2] STT and TTS API Integration...")
    
    # 1. TTS: Text to Speech
    print("TTS: Generating audio...")
    text = "The quick brown fox jumps over the lazy dog."
    tts_payload = {
        "input": text,
        "voice": "af_heart",
        "model": "openai/hexgrad/Kokoro-82M"
    }
    response = await client.post("/v1/openai/audio/speech", json=tts_payload)
    assert response.status_code == 200
    audio_content = response.content
    assert len(audio_content) > 1000
    print(f"Audio generated: {len(audio_content)} bytes")

    # 2. STT: Speech to Text
    print("STT: Transcribing audio...")
    files = {"file": ("test.mp3", audio_content, "audio/mpeg")}
    data = {"model": "openai/openai/whisper-large-v3-turbo"}
    
    response = await client.post("/v1/openai/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200
    transcription = response.json()["text"]
    print(f"Transcription: {transcription}")
    assert transcription.lower() == text.lower()

@pytest.mark.anyio
async def test_3_rag_tools_stream(client):
    print("\n[Test 3] RAG and Tools (Stream)...")
    
    # Using Qwen-4B with internal tools
    # 1. Save info to memory
    secret_info = f"O código secreto de hoje é JULLY-{int(time.time())}"
    save_prompt = f"Grave na minha memória que {secret_info}"
    
    print(f"Saving to memory: {secret_info}")
    payload = {
        "model": "Qwen-4B",
        "messages": [{"role": "user", "content": save_prompt}],
        "stream": True
    }
    
    async with client.stream("POST", "/v1/openai/chat/completions", json=payload, headers={"x-enable-internal-mcp": "1"}) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            pass # Just consume the stream
            
    # 2. Retrieve info from memory
    print("Retrieving from memory...")
    retrieve_prompt = "Qual é o meu código secreto de hoje?"
    payload["messages"] = [{"role": "user", "content": retrieve_prompt}]
    
    full_response = ""
    async with client.stream("POST", "/v1/openai/chat/completions", json=payload, headers={"x-enable-internal-mcp": "1"}) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                data = json.loads(line[6:])
                if data.get("choices"):
                    full_response += data["choices"][0]["delta"].get("content", "") or ""
                    
    print(f"Retrieval Response: {full_response}")
    assert "JULLY-" in full_response
