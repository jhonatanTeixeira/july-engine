import sys
import os
import json
import asyncio
from typing import List, Dict, Any

# Mocking FastAPI/Uvicorn environment enough to test the handler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from july_engine.routers.models import model_warmup, WarmupRequest, WarmupItem
from july_engine.model_loader import model_loader

async def test_warmup():
    print("--- Inciando Teste de Warmup ---")
    
    # Vamos testar o warmup de um modelo pequeno (Qwen3-0.6B-FP16)
    request = WarmupRequest(models=[
        WarmupItem(task_type="brain", model="Qwen3-0.6B-FP16"),
        WarmupItem(task_type="mouth", model="xtts") # XTTS também é pesado
    ])
    
    try:
        result = await model_warmup(request)
        print("Resultado do Warmup:")
        print(json.dumps(result, indent=2))
        
        # Verifica se o modelo está 'is_loaded' no loader
        brain = model_loader.get_brain("gpu", "Qwen3-0.6B-FP16")
        if hasattr(brain, "is_loaded") and brain.is_loaded():
            print("SUCESSO: Brain Qwen3-0.6B-FP16 carregado.")
        else:
            print("AVISO: Brain Qwen3-0.6B-FP16 não reportou is_loaded=True (pode ser delay ou erro silencioso).")

        mouth = model_loader.get_mouth("gpu", "xtts")
        if hasattr(mouth, "is_loaded") and mouth.is_loaded():
            print("SUCESSO: Mouth XTTS carregado.")
        else:
            print("AVISO: Mouth XTTS não reportou is_loaded=True.")

    except Exception as e:
        print(f"ERRO durante teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_warmup())
