"""OCR sobre imagem solta (não PDF) -- ex.: foto de documento fotografado
pra verificação de idade (age-verification pipeline, dating_v3/july-verify).
Reaproveita `_ocr_region` (pdf_extractor.py) pra não duplicar a config real
do Tesseract (upscale 2x, psm 6, lang pt), e `TextDetector` (vision.py,
YOLOv11n-text) pra localizar regiões de texto antes de rodar OCR nelas."""

from PIL import Image

from .pdf_extractor import _ocr_region
from .vision import get_text_detector

Box = tuple[int, int, int, int]


def _group_into_lines(boxes: list[Box]) -> list[list[Box]]:
    """`TextDetector` devolve caixas na ordem de detecção/confiança do
    YOLO, não na ordem espacial de leitura -- concatenar nessa ordem crua
    embaralha palavras de linhas diferentes (achado real: duas linhas de
    um documento saíam intercaladas no `full_text`). Agrupa por proximidade
    vertical (uma "linha" = caixas cujo centro Y cai dentro de uma janela
    baseada na altura média das caixas), devolve as linhas ordenadas de
    cima pra baixo, cada uma já ordenada da esquerda pra direita."""
    if not boxes:
        return []

    avg_height = sum(y2 - y1 for _, y1, _, y2 in boxes) / len(boxes)
    line_threshold = avg_height * 0.6

    lines: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        y_center = (box[1] + box[3]) / 2
        if lines:
            current_line = lines[-1]
            current_line_y = sum((b[1] + b[3]) / 2 for b in current_line) / len(current_line)
            if abs(y_center - current_line_y) <= line_threshold:
                current_line.append(box)
                continue
        lines.append([box])

    return [sorted(line, key=lambda b: b[0]) for line in lines]


def ocr_image(image: Image.Image, *, lang: str = "por", detect_regions: bool = True) -> dict:
    """`detect_regions=True` (default) localiza regiões de texto primeiro,
    agrupa em linhas (ordem de leitura) e roda OCR região por região -- mais
    preciso que a imagem inteira de uma vez, especialmente numa foto de
    documento com fundo/ruído ao redor do texto. `detect_regions=False`
    roda OCR direto na imagem toda."""
    if not detect_regions:
        return {"regions": [], "full_text": _ocr_region(image, lang=lang)}

    lines = _group_into_lines(get_text_detector().detect_text_regions(image))
    regions = []
    text_lines = []
    for line in lines:
        words = []
        for box in line:
            text = _ocr_region(image.crop(box), lang=lang)
            if text:
                regions.append({"bbox": list(box), "text": text})
                words.append(text)
        if words:
            text_lines.append(" ".join(words))

    return {"regions": regions, "full_text": "\n".join(text_lines)}
