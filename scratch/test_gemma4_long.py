import sys
import os
import ctypes
import time
sys.path.insert(0, "./vendor/llama-cpp-python")
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

def test():
    model_path = "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q4_K_M.gguf"
    clip_model_path = "/home/jhonatanteixeira/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/653803f092503c04a65164346f3208a36e707693/mmproj-F16.gguf"

    chat_handler = Gemma4ChatHandler(
        clip_model_path=clip_model_path,
        verbose=True,
        use_gpu=True
    )
    
    # We set n_ctx=1024, n_batch=128
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=1024,
        n_batch=128,
        n_gpu_layers=99,
        flash_attn_type=0,
        verbose=True
    )
    
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///home/jhonatanteixeira/.gemini/antigravity/brain/cd865227-899e-41a3-b0c0-bcd00b01925e/media__1777925725189.png"}},
            {"type": "text", "text": "This is a prompt. " * 30}
        ]}
    ]

    print("Generating...")
    res = llm.create_chat_completion(
        messages=messages,
        max_tokens=20,
        temperature=0.0
    )
    
    # Add multiple messages to cumulatively exceed 1024!
    for i in range(3):
        messages.append(res["choices"][0]["message"])
        messages.append({"role": "user", "content": "Tell me more! " * 30})
        print(f"Generating turn {i+2}...")
        res = llm.create_chat_completion(
            messages=messages,
            max_tokens=20,
            temperature=0.0
        )
    print("Success!")

if __name__ == "__main__":
    test()
