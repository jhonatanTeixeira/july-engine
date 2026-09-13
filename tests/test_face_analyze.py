import base64
import os

import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from app.bridge import bridge

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "sad_person.jpg")


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await bridge.stop()


@pytest.mark.anyio
@pytest.mark.gpu
async def test_face_analyze_estimates_age_with_liveness_off(client):
    """DeepFace.analyze real, sem mock (com anti_spoofing desligado) --
    prova que o import lazy corrigido em FaceService/o endpoint novo
    funcionam de ponta a ponta contra um rosto de verdade, não um PNG
    sintético. Achado real desta tarefa: a fixture (`sad_person.jpg`, uma
    foto estática/provavelmente recomprimida) É pega pelo anti-spoofing
    quando ligado (ver teste abaixo) -- não dá pra testar estimativa de
    idade E liveness=true na mesma fixture estática, por definição (liveness
    existe justamente pra rejeitar imagem estática)."""
    with open(_FIXTURE_PATH, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    response = await client.post(
        "/july/v1/vision/face/analyze",
        json={"image": image_b64, "anti_spoofing": False},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["age"], int)
    assert 0 < body["age"] < 120


@pytest.mark.anyio
@pytest.mark.gpu
async def test_face_analyze_rejects_spoofed_photo_with_liveness_on(client):
    """Mesma fixture, `anti_spoofing` no padrão (true) -- valida que o
    caminho de rejeição por spoof funciona de ponta a ponta (achado real:
    `DeepFace.analyze` levanta `SpoofDetected` em vez de devolver
    `is_real=False` no resultado; o endpoint trata isso como resultado
    válido, não erro de servidor)."""
    with open(_FIXTURE_PATH, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    response = await client.post(
        "/july/v1/vision/face/analyze",
        json={"image": image_b64},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"age": None, "is_real": False, "antispoof_score": None}


@pytest.mark.anyio
async def test_face_analyze_requires_image_field(client):
    response = await client.post("/july/v1/vision/face/analyze", json={})
    assert response.status_code == 400
