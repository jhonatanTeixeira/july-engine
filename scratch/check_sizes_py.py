import ctypes
import sys
sys.path.insert(0, "./vendor/llama-cpp-python")
from llama_cpp.llama_cpp import llama_context_params

print("Python struct size:", ctypes.sizeof(llama_context_params))
print("Offset cb_eval:", getattr(llama_context_params, "cb_eval").offset)
print("Offset flash_attn_type:", getattr(llama_context_params, "flash_attn_type").offset)
print("Offset samplers:", getattr(llama_context_params, "samplers").offset)
print("Offset n_samplers:", getattr(llama_context_params, "n_samplers").offset)
