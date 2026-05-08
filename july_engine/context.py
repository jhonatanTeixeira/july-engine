# Re-export da lib extraída para july_engine_libs
# O código fonte vive em july_engine_libs/python/llama_gguf/context.py
from llama_gguf.context import request_id_var, acquired_instances_var

__all__ = ["request_id_var", "acquired_instances_var"]
