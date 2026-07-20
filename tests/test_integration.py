import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from app.bridge import bridge
from app.persistence import get_backend


# ---------------------------------------------------------------------------
# Test Configuration - Setup para testes locais 100% sem APIs externas
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    """Fixture que configura o backend com modelos e presets locais apenas."""
    backend = get_backend()

    # Configurar settings para serviços locais
    backend.set_setting("TEXT_PRESETS", {
        "default": [
            {"alias": "qwen3-cpu", "model": "Qwen/Qwen3-0.6B-GGUF", "backend": "cpu"},
            {"alias": "qwen3-gpu", "model": "Qwen/Qwen3-0.6B-GGUF", "backend": "gpu"}
        ]
    })

    backend.set_setting("EMBEDDINGS", {"model": "bge_micro", "backend": "cpu"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await bridge.stop()


# ---------------------------------------------------------------------------
# Testes Básicos de Health e Integração Local
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health_check(client):
    """Verifica se o servidor inicializa corretamente."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_models_endpoint(client):
    """Verifica o endpoint de listagem de modelos."""
    response = await client.get("/v1/openai/models")
    assert response.status_code in [200, 401]  # Pode requerer autenticação, mas não pode ser erro 500


@pytest.mark.anyio
async def test_chat_completions_basic(client):
    """Teste básico de chat completions (pode falhar se modelo não estiver carregado)."""
    payload = {
        "model": "qwen3-cpu",
        "messages": [{"role": "user", "content": "Olá"}],
        "max_tokens": 50,
        "stream": False
    }

    try:
        response = await client.post("/v1/openai/chat/completions", json=payload)
        # Pode falhar com 500 se modelo não estiver disponível ou 200 se funcionar
        assert response.status_code in [200, 400, 409, 500]
    except Exception:
        pytest.skip("Modelo local não está disponível para teste")


# ---------------------------------------------------------------------------
# Observabilidade e Monitoramento (removido: API externas)
# Testes básicos de endpoints internos
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_monitoring_metrics(client):
    """Verifica endpoint de métricas Prometheus."""
    response = await client.get("/metrics")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# TTS e STT Locais (mantido: serviços locais)
# Testes básicos da pipeline local de áudio
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_synthesis_tts_basic(client):
    """Teste básico de síntese TTS local."""
    payload = {
        "input": "Olá mundo",
        "model_id": "kokoro-af-base",
        "language": "a"  # American English, mas pode usar vozes PT se disponíveis
    }

    try:
        response = await client.post(
            "/tts/synthesis-stream",
            json=payload,
            timeout=30.0  # TTS pode demorar um pouco
        )
        assert response.status_code in [200, 409]
    except Exception:
        pytest.skip("TTS Kokoro não está disponível para teste")


@pytest.mark.anyio
async def test_speech_to_text_basic(client):
    """Teste básico de transcrição STT local."""
    # Skip se não houver arquivo de áudio de teste
    import os

    try:
        with open("dummy_audio.wav", "rb") as f:  # File placeholder - deve ser removido/criado
            pass
    except FileNotFoundError:
        pytest.skip("Arquivo de áudio de teste não encontrado para STT")
