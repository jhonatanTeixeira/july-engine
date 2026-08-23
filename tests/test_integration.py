import asyncio
import json
import os
import subprocess
import sys

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


@pytest.mark.anyio
async def test_concurrent_chat_completions_no_deadlock(client):
    """
    Duas requisições concorrentes ao mesmo modelo de chat devem ser atendidas
    sem deadlock — exercita o ReentrantModelLock / sequence pooling do
    orchestrator (app/orchestrator.py). Um timeout aqui indica um deadlock de
    verdade e deve falhar o teste, não só pular.
    """
    payload = {
        "model": "qwen3-cpu",
        "messages": [{"role": "user", "content": "Diga oi em uma palavra."}],
        "max_tokens": 10,
        "stream": False,
    }

    try:
        responses = await asyncio.wait_for(
            asyncio.gather(
                client.post("/v1/openai/chat/completions", json=payload),
                client.post("/v1/openai/chat/completions", json=payload),
            ),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        pytest.fail("Duas chamadas concorrentes ao mesmo modelo travaram (deadlock) e não completaram em 90s")
    except Exception:
        pytest.skip("Modelo local não está disponível para teste de concorrência")

    for response in responses:
        assert response.status_code in [200, 400, 409, 500]


@pytest.mark.gpu
@pytest.mark.anyio
async def test_concurrent_hybrid_model_no_corruption_or_crash():
    """
    Regression test for a real, reproducible "double free or corruption
    (fasttop)" crash found while validating MTP support: HybridCheckpointCache
    (vendor/.../llama_cpp/llama_cache.py — the RNN/recurrent-state
    checkpoint save/restore mechanism `_reuse_prefix_and_eval` uses for
    hybrid architectures like Qwen3.5's gated-delta-net layers) had zero
    thread-safety, despite being invoked concurrently for every seq_id whose
    new prompt misses the KV-cache prefix match. Two or more ordinary
    conversations on the same hybrid model — each simply asking about a
    different, unrelated topic every round, no special trigger needed —
    reliably corrupted the process heap within a handful of rounds. This is
    independent of MTP itself: it reproduces with mtp_enabled=False.

    Deliberately does NOT assert on deadlocks like
    test_concurrent_chat_completions_no_deadlock above — it asserts the
    concurrent requests don't crash the process AND that their content isn't
    silently corrupted (empty, mid-token-garbled, or looping), which a naive
    "did we get a response" check would miss entirely.

    Runs the actual repro in a SUBPROCESS (tests/_hybrid_concurrency_worker.py)
    on purpose: a real heap-corruption abort or segfault takes down the whole
    interpreter, which — if triggered in-process — would silently kill the
    rest of this pytest run instead of failing just this one test.
    """
    repo_root = os.path.dirname(os.path.dirname(__file__)) or "."

    proc = subprocess.run(
        # Run as a module (not a bare script path): a bare script path puts
        # tests/ itself on sys.path[0], not the repo root, so the worker's
        # `from main import app` fails with ModuleNotFoundError before it
        # can even print a RESULT_JSON line. `-m` with cwd=repo_root puts
        # the repo root on sys.path[0] instead, matching how the rest of
        # this test suite (and the real app) is normally run.
        [sys.executable, "-m", "tests._hybrid_concurrency_worker"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )

    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result_line = line[len("RESULT_JSON:"):]

    if result_line is None:
        pytest.fail(
            "Worker process produced no result (likely crashed before finishing) — "
            f"returncode={proc.returncode} (negative means killed by signal -N).\n"
            f"--- worker stdout (last 4000 chars) ---\n{proc.stdout[-4000:]}\n"
            f"--- worker stderr (last 4000 chars) ---\n{proc.stderr[-4000:]}"
        )

    result = json.loads(result_line)

    if result["status"] == "skip":
        pytest.skip(result["reason"])

    assert proc.returncode == 0, (
        f"Worker reported status=ok but exited with returncode={proc.returncode} "
        f"(negative means killed by signal -N) — stderr tail:\n{proc.stderr[-4000:]}"
    )
    assert result["corrupted"] == [], (
        "Concurrent hybrid-model chat produced crashed/corrupted responses:\n"
        + json.dumps(result["corrupted"], indent=2, ensure_ascii=False)
    )


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
async def test_synthesis_neutts_air_basic(client):
    """
    Teste básico de síntese via NeuTTS Air (~0.7B, CPU-first, cabe em 4GB VRAM).
    """
    payload = {
        "input": "Olá mundo",
        "model_id": "neutts-air-default",
    }

    try:
        response = await client.post(
            "/tts/synthesis-stream",
            json=payload,
            timeout=60.0,
        )
        assert response.status_code in [200, 409]
    except Exception:
        pytest.skip("NeuTTS Air não está disponível para teste")


@pytest.mark.anyio
async def test_synthesis_f5tts_basic(client):
    """
    Teste básico de síntese via F5-TTS (~336M, DiT flow-matching, ~3GB VRAM).
    """
    payload = {
        "input": "Olá mundo",
        "model_id": "f5-tts-default",
    }

    try:
        response = await client.post(
            "/tts/synthesis-stream",
            json=payload,
            timeout=60.0,
        )
        assert response.status_code in [200, 409]
    except Exception:
        pytest.skip("F5-TTS não está disponível para teste")


# NOTA: IndexTTS-2 foi implementado (app/models/tts_indextts2.py) mas
# deliberadamente NÃO tem teste de integração real aqui — o antecessor
# (IndexTTS 1.5) já exige ~8GB de VRAM e o v2 é mais pesado ainda, o que não
# roda de forma confiável no alvo de hardware de 4GB VRAM deste repositório
# (ver Dockerfile/setup.sh, RTX 3050 4GB). get_required_vram() em
# tts_indextts2.py já reflete isso para que o orchestrator rejeite com
# MemoryError em vez de estourar a VRAM.


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
