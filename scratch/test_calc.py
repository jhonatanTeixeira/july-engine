import asyncio
import os
from jully_engine.services.resource_calculator import estimate_vram_ram

async def main():
    # Test with a mock path that exists in cache
    # Qwen3.6-35B-A3B: storage/cache/7d9e9de6808a2da4205f2977c7e61678.json
    # We need to pass a path that md5 to 7d9e9de6808a2da4205f2977c7e61678
    # Or just use a real file if possible, but let's just point to a valid metadata path if we can hack it.
    
    # Actually, ModelMetadata md5s the path.
    # Let's find what path generates that md5.
    # Wait, I don't know the path.
    
    # Let's test with the gemma4 metadata.
    # Path: /home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf
    path = "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf"
    
    res = await estimate_vram_ram(
        path,
        context_window=8192,
        n_seq_max=2,
        kv_cache_quantization="Q8_0"
    )
    print("Result:", res)

asyncio.run(main())
