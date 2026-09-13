import base64
import io

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image, ImageDraw, ImageFont

from main import app
from app.bridge import bridge

_FACE_FIXTURE_PATH = "tests/sad_person.jpg"
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await bridge.stop()


def _synthetic_document_photo() -> bytes:
    img = Image.new("RGB", (640, 200), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(_FONT_PATH, 32)
    draw.text((20, 40), "NOME: MARIA DA SILVA", fill="black", font=font)
    draw.text((20, 100), "DATA DE NASCIMENTO: 15/03/1990", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.anyio
@pytest.mark.gpu
async def test_deepface_then_yolo_in_same_process(client):
    """Reproduz, em pytest puro (sem servidor solto/AgentGateway), a mesma
    sequência que o july-verify faz de verdade: uma chamada DeepFace
    (TensorFlow) seguida de uma chamada YOLO/OCR (PyTorch) no MESMO
    processo -- isolando se o conflito é de fato TF->PyTorch ou outra
    causa."""
    with open(_FACE_FIXTURE_PATH, "rb") as fh:
        face_b64 = base64.b64encode(fh.read()).decode("utf-8")

    face_response = await client.post("/july/v1/vision/face/embedding", json={"image": face_b64})
    print("FACE RESPONSE:", face_response.status_code, face_response.text[:200])

    ocr_response = await client.post(
        "/july/v1/utils/ocr-image",
        files={"file": ("doc.png", _synthetic_document_photo(), "image/png")},
        data={"lang": "por", "detect_regions": "true"},
    )
    print("OCR RESPONSE:", ocr_response.status_code, ocr_response.text[:500])

    assert face_response.status_code == 200, face_response.text
    assert ocr_response.status_code == 200, ocr_response.text
