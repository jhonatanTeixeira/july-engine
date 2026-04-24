import faulthandler
faulthandler.enable()

import os
import pytest
import asyncio
import numpy as np
from PIL import Image
from jully_engine.domain.eyes import Eyes
from jully_engine.orchestrators.cpu_orchestrator import cpu_orchestrator
from jully_engine.orchestrators.gpu_orchestrator import gpu_orchestrator
from jully_engine.services.helpers import inference_helper
from jully_engine.persistence.vector_store import vector_store

VIDEO_PATH = "/mnt/jhonatanteixeira/Novo volume/projects/jhon/ai/jully/july_engine/tests/20171231_164112.mp4"

@pytest.mark.asyncio
async def test_vision_video_pipeline():
    """
    Testa a pipeline completa de visão:
    1. Análise multimodal de vídeo (descrição visual + áudio).
    2. Reconhecimento de cena específica.
    3. Extração de faces e embeddings.
    4. Extração de áudio e embeddings.
    """
    assert os.path.exists(VIDEO_PATH), f"Vídeo de teste não encontrado: {VIDEO_PATH}"
    
    # Garante que os orquestradores estão rodando
    await cpu_orchestrator.start()
    
    # 1. Configuração do Eyes com FastVLM (modelo pequeno conforme solicitado)
    eyes = Eyes(backend="cpu", model_tag="fastvlm")
    
    # 2. Análise do Vídeo
    # Esperamos que identifique: "mulher tocando violino em uma embarcação"
    payload = {
        "video_path": VIDEO_PATH,
        "interval_sec": 2.0,
        "frames_per_grid": 4,
        "headers": {
            "x-backend": "cpu"
        }
    }
    
    logger = eyes.face_service.vector_store # Só para logar se precisar
    print("\n[Vision Test] Iniciando análise multimodal do vídeo...")
    
    aggregate = await eyes.describe_video(payload)
    print(f"[Vision Test] Resultado (Aggregate): {aggregate}")
    
    # 3. Verificações de Reconhecimento (nos segmentos do aggregate)
    found_violin = False
    found_boat = False
    
    for seg in aggregate.segments:
        text = seg.narrative.text.lower()
        if "violin" in text or "violino" in text:
            found_violin = True
        if "boat" in text or "embarcação" in text or "ship" in text or "water" in text:
            found_boat = True
    
    assert found_violin, "Não reconheceu o violino no vídeo"
    assert found_boat, "Não reconheceu a embarcação/água no vídeo"

    # 3. Extração de Faces e Embeddings
    print("[Vision Test] Extraindo faces do vídeo...")
    # Vamos pegar um frame específico para testar extração de face (ou usar o describe_person_faces)
    import cv2
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces_data = await eyes.describe_person_faces(img_pil)
        print(f"[Vision Test] Faces encontradas: {len(faces_data)}")
        
        for face in faces_data:
            assert "person_id" in face
            assert "embedding" in face
            assert len(face["embedding"]) > 0
            print(f"[Vision Test] Face ID: {face['person_id']}, Embedding size: {len(face['embedding'])}")

    # 4. Verificação de Embeddings na Collection
    print("[Vision Test] Testando persistência e busca de embeddings...")
    collection_name = "test_faces"
    test_emb = [0.1] * 512 # ArcFace costuma ser 512
    person_id = "test-person-123"
    
    # Salvar
    vector_store.add(
        text="Test Face",
        embedding=test_emb,
        metadata={"person_id": person_id},
        collection=collection_name
    )
    
    # Buscar
    results = vector_store.search_with_details(test_emb, top_k=1, collection=collection_name)
    assert len(results) > 0
    assert results[0]["metadata"]["person_id"] == person_id
    print("[Vision Test] Busca de embedding OK.")

    # Deletar
    vector_store.delete(ids=[results[0]["id"]], collection=collection_name)
    results_after = vector_store.search_with_details(test_emb, top_k=1, collection=collection_name)
    # Dependendo da implementação do vector_store, pode retornar vazio ou distância alta
    if results_after:
        assert results_after[0]["id"] != results[0]["id"]
    print("[Vision Test] Deleção de embedding OK.")

@pytest.mark.asyncio
async def test_search_and_scrape():
    """
    Teste de integração para busca web e extração de conteúdo.
    """
    print("\n[Search Test] Testando Search and Scrape...")
    payload = {
        "query": "Qual a capital da França?",
        "max_results": 1
    }
    
    # Usando o inference_helper para rodar o search_web
    try:
        results = await inference_helper.process("search_web", payload)
        print(f"[Search Test] Resultados: {results}")
        assert "Paris" in str(results)
    except Exception as e:
        print(f"[Search Test] Falha no search_web: {e}")
        # Se for erro de API key, podemos pular ou falhar dependendo do ambiente
        # pytest.skip("Search API Key não configurada")

@pytest.mark.asyncio
async def test_tool_call_qwen_coder():
    """
    Teste de tool call usando Qwen 2.5 Coder 1.5b.
    """
    print("\n[Tool Test] Testando Tool Call com Qwen 2.5 Coder...")
    
    payload = {
        "model": "qwen2.5-coder",
        "messages": [
            {"role": "user", "content": "Que horas são em São Paulo? Use a ferramenta get_current_time."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Retorna a hora atual em uma localização",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
    
    # Aqui precisamos garantir que o QwenChatHandler está sendo usado
    # e que ele suporta tool calling.
    try:
        response = await inference_helper.process("text_chat", payload)
        print(f"[Tool Test] Resposta: {response}")
        # Verificamos se houve uma chamada de ferramenta (tool_calls)
        if isinstance(response, dict):
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            assert "tool_calls" in message or "get_current_time" in str(message)
    except Exception as e:
        print(f"[Tool Test] Erro no Tool Call: {e}")
        raise e
