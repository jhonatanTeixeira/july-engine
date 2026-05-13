import os
import pprint

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pytest
import json
import base64
from httpx import AsyncClient, ASGITransport

# Force the environment to test
os.environ['ENV'] = 'test'
os.environ['PERSISTENCE_BACKEND'] = 'tinydb'
os.environ['DB_PATH'] = 'testes.json'

@pytest.fixture(scope="module")
def anyio_backend():
    return 'asyncio'

@pytest.fixture(scope="module")
async def client():
    # Import inside fixture to ensure env vars are set
    from main import app
    from app.bridge import bridge
    from app.persistence.tinydb_backend import TinyDBBackend
    from app.persistence import persistence
    
    # Use a specific test database named 'testes.json'
    test_db_path = os.path.join("storage", "db", "testes.json")
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    backend = TinyDBBackend(test_db_path)
    persistence._backend_instance = backend
    
    # backend = persistence.get_backend()
    
    # --- Inject Settings ---
    backend.set_setting("STT", {"model": "faster-whisper", "backend": "gpu"})
    backend.set_setting("TTS", {"model": "kokoro", "backend": "gpu", "voice": "af_sky", "language": "a"})
    backend.set_setting("VISION", {"model": "fastvlm", "backend": "gpu"})
    backend.set_setting("IMAGE_CREATE", {"model": "lcm", "backend": "gpu"})
    # backend.set_setting("FLUX", {"model": "flux-klein", "backend": "gpu"})
    backend.set_setting("REMBG", {"model": "rembg", "backend": "gpu"})
    backend.set_setting("WEB_SEARCH", {"model": "tavily", "backend": "api"})
    backend.set_setting("EMBEDDINGS", {"model": "bge_micro", "backend": "cpu"})
        
    text_presets = [
        {"alias": "qwen3-cpu", "model": "qwen3-0.6b", "backend": "cpu"},
        {"alias": "qwen3-gpu", "model": "qwen3-0.6b", "backend": "gpu", "mcp_option":  "emulated",},
        {"alias": "qwen3-gpu-mcp", "model": "qwen3-0.6b", "backend": "gpu", "mcp_option": "emulated"},
        {
            "alias": "Qwen3.5-0.8B",
            "model": "Qwen3.5-0.8B",
            "api_key": "",
            "backend": "gpu",
            "base_url": "",
            "is_vision": True,
            "is_default": False,
            "mcp_option": "internal"
        },
          {
            "alias": "Qwen3.5-4B",
            "model": "Qwen3.5-4B",
            "api_key": "",
            "backend": "gpu",
            "base_url": "",
            "is_vision": False,
            "is_default": False,
            "mcp_option": "internal"
        },
    ]

    backend.set_setting("TEXT_PRESETS", text_presets)
    
    # --- Inject Models ---
            
    backend.set_model("qwen3-0.6b", {
        "model_alias": "Qwen3-0.6B",
        "model_type": "text",
        "model_id": "bartowski/Qwen_Qwen3-0.6B-GGUF",
        "filename": "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        "mmproj_id": None,
        "mmproj_filename": None,
        "template": "qwen",
        "context_window": 2048,
        "num_params": 0.6,
        "quantization": "Q4_K_M",
        "num_layers": -1,
        "force_reasoning": None,
        "file_path": "C:\\Users\\jhona/.cache/huggingface/hub\\models--appleyu--Qwen3-0.6B-FP16-gguf\\snapshots\\421187a1573b0ac2be5466d7b45da087c5ee3367\\Qwen3-0.6B-FP16.gguf",
        "mmproj_path": None
    })

    backend.set_model("Qwen3.5-0.8B", {
        "filename": "Qwen3.5-0.8B-Q4_K_M.gguf", 
        "model_id": "unsloth/Qwen3.5-0.8B-GGUF", 
        "template": "qwen", 
        "file_path": "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf", 
        "is_vision": True, 
        "mmproj_id": "unsloth/Qwen3.5-0.8B-GGUF", 
        "n_seq_max": 1, 
        "flash_attn": True, 
        "kv_unified": False, 
        "logits_all": False, 
        "model_type": "vision", 
        "num_layers": -1, 
        "mmproj_path": "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/mmproj-F16.gguf", 
        "model_alias": "Qwen3.5-0.8B", 
        "offload_kqv": False, 
        "context_window": 8192, 
        "force_reasoning": True, 
        "mmproj_filename": "mmproj-F16.gguf", 
        "kv_cache_quantization": "Q8_0"
    })

    backend.set_model("Qwen3.5-4B",	{
        "filename": "Qwen3.5-4B-Q4_K_M.gguf", 
        "model_id": "unsloth/Qwen3.5-4B-GGUF", 
        "template": "qwen", 
        "file_path": "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf", 
        "is_vision": True, 
        "mmproj_id": "unsloth/Qwen3.5-4B-GGUF", 
        "n_seq_max": 1, 
        "flash_attn": True, 
        "kv_unified": False, 
        "logits_all": False, 
        "model_type": "vision", 
        "num_layers": -1, 
        "mmproj_path": "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/mmproj-F16.gguf",
        "model_alias": "Qwen3.5-4B", 
        "offload_kqv": False, 
        "vision_on_cpu": True,
        "context_window": 12288,
        "force_reasoning": True, 
        "mmproj_filename": "mmproj-F16.gguf", 
        "kv_cache_quantization": "Q8_0"
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
    reasoning_content = response.json()["choices"][0]["message"].get("reasoning_content", "").strip()

    print(f"Chat Output: {content}")
    print(f"Reasoning Chat Output: {reasoning_content}")
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
        "model": "qwen3-gpu",
        "messages": [
            {"role": "user", "content": "olá, tudo bem?"}
        ],
        "stream": True
    }
    
    reasoning_found = False
    content_found = False
    headers = {"x-backend": "gpu"}

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
        "model": "qwen3-gpu",
        "messages": [
            {"role": "user", "content": "olá, tudo bem?"}
        ],
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
        "model": "Qwen3.5-0.8B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with image generation capabilities. When asked to generate an image, you MUST use the generate image tool"},
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

    import re
    
    if isinstance(content, list):
        assert any([part.get("type") == "image_url" and re.match(r'data:image/\w+;base64,.*', part.get("image_url").get("url")) for part in content]), f"Image was not generated in non-stream MCP mode. Content received: {content}"

    elif isinstance(content, str):
        assert re.match(r'data:image/\w+;base64,.*', content), f"Image was not generated in non-stream MCP mode. Content received: {content}"
        
    print("Non-Stream MCP Image Generation: OK")

    # 2. Stream Mode
    print("Testing Stream MCP...")
    payload_stream = {
        "model": "qwen3-gpu-mcp",
        "messages": [
            # {"role": "system", "content": "You are a helpful assistant with image generation capabilities. When asked to generate an image, use the <generate_image><prompt>...</prompt></generate_image> tool."},
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
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                
                if delta.get('type', None) == "image_url":
                    assert re.match(r'data:image/\w+;base64,.*', delta.get("image_url")), f"Image was not generated in stream MCP mode. Content received: {content}"
                    stream_image_found = True
                    print("[Image Chunk Found]", flush=True)
    
    assert stream_image_found, "Image was not generated in stream MCP mode"
    print("Stream MCP Image Generation: OK")

# --- Video Description Integration ---
@pytest.mark.anyio
async def test_video_description_strategies(client):
    print("\n[Test] Running Video Description Strategy Integration...")
    video_path = "tests/20171231_164112.mp4"

    strategies = ["default", "interaction", "emotion"]

    
    for strategy in strategies:
        print(f"Testing strategy: {strategy}")
        with open(video_path, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            data = {
                "interval_sec": "1.0",
                "frames_per_grid": "1",
                "strategy": strategy,
                "model": "fastvlm",
                "description_model": "qwen3-cpu"
            }
            # Headers for the backend
            headers = {
                "x-backend": "cpu",
                "x-context-window": "32768"
            }
            
            response = await client.post("/july/v1/vision/video/describe", files=files, data=data, headers=headers)
            
            if response.status_code != 200:
                print(f"error {response.status_code} {response.reason_phrase}")
                pprint.pprint(response.json())
                raise AssertionError(response)
            
            res_json = response.json()
            assert "visual_narrative" in res_json
            narrative = res_json["visual_narrative"]
            print(f"Strategy {strategy} narrative snippet: {narrative[:100]}...")
            assert len(narrative) > 0
    
    print("Video Description Strategies: OK")

# --- RAG Integration ---
@pytest.mark.anyio
async def test_rag_integration(client):
    print("\n[Test] Running RAG Integration...")
    collection = "test_collection"
    text = "The quick brown fox jumps over the lazy dog."
    
    # 1. Add to RAG
    add_payload = {
        "text": text,
        "collection": collection,
        "metadata": {"source": "test_fox"}
    }
    add_response = await client.post("/july/v1/rag", json=add_payload)
    if add_response.status_code != 200:
        print(f"Error Response: {add_response.text}")
    assert add_response.status_code == 200
    assert add_response.json()["success"] is True

    print('add response', add_response)
    
    # 2. Search in RAG
    search_payload = {
        "query": "lazy dog",
        "collection": collection,
        "top_k": 1
    }
    search_response = await client.post("/july/v1/rag/search", json=search_payload)
    assert search_response.status_code == 200
    
    results = search_response.json()
    print(f"RAG search results: {results}")
    assert len(results) > 0
    # Check if the text matches (RagAdapter search returns results with 'content' or 'text')
    found_text = results[0].get("content") or results[0].get("text")
    print(f"RAG search result: {found_text}")
    assert text in found_text
    print("RAG Integration: OK")

# --- Multimodal Complex Orchestration ---
@pytest.mark.anyio
async def test_multimodal_complex_orchestration(client):
    print("\n[Test] Running Multimodal Complex Orchestration...")
    headers_gpu = {"x-backend": "gpu"}
    
    # 1. Generate Image with Flux-Klein
    print("Step 1: Generating image with flux-klein (GPU)...")
    gen_payload = {
        "model": "flux-klein",
        "prompt": "Full body photo of a beautiful woman standing in a vibrant flower garden, colorful background, 8k, highly detailed",
        "response_format": "b64_json"
    }
    gen_response = await client.post("/v1/openai/images/generations", json=gen_payload, headers=headers_gpu)
    assert gen_response.status_code == 200, f"Generation failed: {gen_response.text}"
    img_b64 = gen_response.json()["data"][0]["b64_json"]
    print("Image generated successfully.")

    # 2. Remove Background
    print("Step 2: Removing background (rembg)...")
    img_bytes = base64.b64decode(img_b64)
    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"model": "rembg"}
    
    # rembg can run on gpu or cpu, depending on config, here we use gpu headers to test orchestrator
    bg_response = await client.post("/july/v1/vision/images/remove-background", files=files, data=data, headers=headers_gpu)
    assert bg_response.status_code == 200, f"Background removal failed: {bg_response.text}"
    img_no_bg_b64 = bg_response.json()["image"]
    print("Background removed successfully.")

    # 3. Verify with Qwen3.5-0.8B (Multimodal)
    print("Step 3: Verifying transparency with Qwen3.5-4B (GPU)...")
    vision_payload = {
        "model": "Qwen3.5-4B",
        "messages": [
            {"role": "system", "content": "Você é um assistente visual preciso. Responda de forma curta e direta."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Observe esta imagem. O fundo está transparente ou foi removido, restando apenas a pessoa? Responda apenas 'sim' ou 'não'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_no_bg_b64}"}}
                ]
            }
        ]
    }
    
    vision_response = await client.post("/v1/openai/chat/completions", json=vision_payload, headers=headers_gpu)
    assert vision_response.status_code == 200, f"Vision verification failed: {vision_response.text}"
    
    answer = vision_response.json()["choices"][0]["message"]["content"].strip().lower()
    print(f"Qwen3.5-4B Answer: {answer}")
    
    # We expect 'sim' or something positive
    assert "sim" in answer or "yes" in answer or "transparente" in answer

    # 2. Remove Background
    print("Step 4: Editing image")
    img_bytes = base64.b64decode(img_no_bg_b64)
    files = {"image": ("image.png", img_bytes, "image/png")}
    data = {"model": "flux-klein"}
    
    edit_response = await client.post("/v1/openai/images/edits", files=files, data=data, headers=headers_gpu)
    assert edit_response.status_code == 200, f"Background removal failed: {bg_response.text}"
    edit_b64 = edit_response.content
    print("Image edited successfully.")

    print("Step 5: Verifying image edited with Qwen3.5-4B (GPU)...")
    vision_payload = {
        "model": "Qwen3.5-4B",
        "messages": [
            {"role": "system", "content": "Você é um assistente visual preciso. Responda de forma curta e direta."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A pessoa na imagem está triste? Responda apenas 'sim' ou 'não'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edit_b64}"}}
                ]
            }
        ]
    }
    
    vision_response = await client.post("/v1/openai/chat/completions", json=vision_payload, headers=headers_gpu)
    assert vision_response.status_code == 200, f"Vision verification failed: {vision_response.text}"
    
    answer = vision_response.json()["choices"][0]["message"]["content"].strip().lower()
    print(f"Qwen3.5-4B Answer: {answer}")
    
    # We expect 'sim' or something positive
    assert "sim" in answer or "yes" in answer or "transparente" in answer

    print("Multimodal Complex Orchestration: OK")
