import argparse
import json
import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="Test Llama.cpp Python with Tool Calling")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace Repository ID (e.g., NousResearch/Hermes-3-Llama-3.1-8B-GGUF)")
    parser.add_argument("--filename", type=str, required=True, help="Filename of the GGUF model (e.g., Hermes-3-Llama-3.1-8B.Q4_K_M.gguf)")
    parser.add_argument("--chat-format", type=str, default="chatml-function-calling", help="Chat format (e.g., chatml, chatml-function-calling)")
    args = parser.parse_args()

    print(f"Checking cache or downloading: {args.repo_id} / {args.filename}...")
    
    # Use HF Hub to find or download the model
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    try:
        model_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            cache_dir=cache_dir
        )
        print(f"Model found at: {model_path}")
    except Exception as e:
        print(f"❌ Failed to download or locate model: {e}")
        return

    print(f"\nLoading model into Llama.cpp with format '{args.chat_format}'...")
    
    # Initialize the model
    llm = Llama(
        model_path=model_path,
        chat_format=args.chat_format,
        n_ctx=2048,
        verbose=False
    )

    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get the current temperature for a city",
                "parameters": {
                    "type": "object",
                    "required": ["city"],
                    "properties": {
                        "city": {"type": "string", "description": "The name of the city"}
                    }
                },
            }
        }
    ]

    messages = [
        # {"role": "system", "content": "you have access to get_temperature function, if the user asks for temperature or wheater, call this tool"},
        # {"role": "system", "content": "You are a helpful assistant. if the user wnats to know the wheter call the get_current_weather."},
        {"role": "user", "content": "whats the current temperature on new york?"}
    ]

    print("\nSending prompt to LLM...")
    
    # Call the model
    try:
        response = llm.create_chat_completion(
            messages=messages,
            tools=tools,
            # tool_choice={"type": "function", "function": {"name": "get_temperature"}}, # Force the tool
            temperature=0.1
        )

        print("\n--- Response ---")
        print(json.dumps(response, indent=2))
        
        # Check if a tool was called
        message = response['choices'][0]['message']
        if 'tool_calls' in message and message['tool_calls']:
            print("\n✅ Success! The model called the tool:")
            for tool_call in message['tool_calls']:
                print(f"Tool Name: {tool_call['function']['name']}")
                print(f"Arguments: {tool_call['function']['arguments']}")
        else:
            print("\n❌ Failed: The model did not return any tool calls.")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")

if __name__ == "__main__":
    main()
