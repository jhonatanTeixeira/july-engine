import io

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image, ImageDraw, ImageFont

from main import app
from app.bridge import bridge

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await bridge.stop()


def _synthetic_document_photo() -> bytes:
    """Não é uma foto de documento real (nunca simular/usar um RG de
    verdade ou verossímil em teste) -- imagem sintética com texto renderizado
    de verdade via PIL, pixels reais pro pipeline real (detecção + OCR)
    processar, só não é uma fotografia de um objeto físico."""
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
async def test_ocr_image_reads_text_with_region_detection(client):
    """Pipeline real, sem mock: TextDetector (YOLOv11n-text) localiza as
    regiões de texto, OCR (pytesseract, via _ocr_region reaproveitado do
    pdf_extractor) lê cada uma -- prova as duas primitivas novas (detecção +
    OCR de imagem solta) de ponta a ponta."""
    files = {"file": ("doc.png", _synthetic_document_photo(), "image/png")}

    response = await client.post(
        "/july/v1/utils/ocr-image",
        files=files,
        data={"lang": "por", "detect_regions": "true"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["regions"]) >= 1
    full_text = body["full_text"].upper()
    assert "1990" in full_text
    assert "NASCIMENTO" in full_text
    # Ordem de leitura, não só presença das palavras -- achado real: sem
    # agrupar por linha, o YOLO devolve as caixas em ordem de
    # detecção/confiança, embaralhando as duas linhas do documento.
    assert full_text.index("NOME") < full_text.index("SILVA") < full_text.index("DATA")
    assert full_text.index("NASCIMENTO") < full_text.index("1990")


@pytest.mark.anyio
@pytest.mark.gpu
async def test_ocr_image_without_region_detection(client):
    """`detect_regions=false` -- OCR direto na imagem toda, sem o
    TextDetector no caminho (fallback mais simples)."""
    files = {"file": ("doc.png", _synthetic_document_photo(), "image/png")}

    response = await client.post(
        "/july/v1/utils/ocr-image",
        files=files,
        data={"lang": "por", "detect_regions": "false"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["regions"] == []
    assert "1990" in body["full_text"]
