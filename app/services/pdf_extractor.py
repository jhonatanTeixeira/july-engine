"""
PDF Extraction Pipeline
=======================
Extrai texto e imagens de PDFs de livros ilustrados (e similares), com:

1. Extração de texto nativo do PDF (se houver camada de texto)
2. Detecção automática do layout: split illustration | text
3. OCR na região de texto (Tesseract, lang=por)
4. Crop limpo da ilustração (sem texto)
5. Inpainting opcional para remover texto residual da ilustração

Dependências:
    pip install pymupdf pytesseract opencv-python-headless Pillow numpy
    apt install tesseract-ocr tesseract-ocr-por
"""

import io
import base64
import logging
import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PageImage:
    xref: int
    width: int
    height: int
    ext: str
    image_base64: str          # data URI  data:image/png;base64,...


@dataclass
class PageResult:
    page: int
    native_text: str           # texto da camada PDF (vazio se não houver)
    ocr_text: str              # texto extraído via OCR da região de texto
    illustration: Optional[str] = None   # data URI da ilustração recortada
    raw_images: list[PageImage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def _render_page(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    """Rasteriza a página inteira em alta resolução."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _extract_native_text(page: fitz.Page) -> str:
    """Extrai texto da camada vetorial do PDF (vazio em PDFs rasterizados)."""
    text = ""
    for block in page.get_text("blocks"):
        if block[6] == 0:   # block_type 0 = texto
            text += block[4] + "\n"
    return text.strip()


def _extract_raw_images(doc: fitz.Document, page: fitz.Page) -> list[PageImage]:
    """
    Extrai todas as imagens embutidas na página (excluindo smasks).
    Converte CMYK → RGB quando necessário.
    """
    imgs = page.get_images(full=True)
    smask_xrefs = {img[1] for img in imgs if img[1] != 0}
    results = []

    for img in imgs:
        xref = img[0]
        if xref in smask_xrefs:
            continue

        base = doc.extract_image(xref)
        if not base or "image" not in base:
            continue

        image_bytes = base["image"]
        ext = base.get("ext", "png")

        # Converter CMYK → RGB
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            image_bytes = pix.tobytes("png")
            ext = "png"

        b64 = base64.b64encode(image_bytes).decode()
        results.append(PageImage(
            xref=xref,
            width=base.get("width", img[2]),
            height=base.get("height", img[3]),
            ext=ext,
            image_base64=f"data:image/{ext};base64,{b64}",
        ))

    return results


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_split_x(arr: np.ndarray) -> int:
    """
    Detecta o ponto de divisão vertical entre ilustração (esquerda)
    e região de texto (direita) procurando a coluna com menor variância
    de cor na metade central da página.

    Retorna o índice x do split (em pixels do array).
    """
    h, w = arr.shape[:2]
    center_start = w // 3
    center_end = 2 * w // 3

    col_vars = [
        arr[:, x, :].std()
        for x in range(center_start, center_end, 4)
    ]
    min_rel = int(np.argmin(col_vars))
    split_x = center_start + min_rel * 4
    return split_x


def _detect_layout(composite: Image.Image) -> dict:
    """
    Retorna {split_x, is_spread}.
    is_spread = True quando a página tem layout spread (illus | texto).
    Heurística: se a diferença de variância entre metade esquerda e direita
    for grande, é um spread; caso contrário é uma página simples.
    """
    arr = np.array(composite)
    h, w = arr.shape[:2]
    mid = w // 2

    left_var = arr[:, :mid, :].std()
    right_var = arr[:, mid:, :].std()
    diff = abs(float(left_var) - float(right_var))

    if diff > 15:   # threshold empírico
        split_x = _detect_split_x(arr)
        return {"is_spread": True, "split_x": split_x}
    else:
        return {"is_spread": False, "split_x": None}


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _ocr_region(img_region: Image.Image, lang: str = "por") -> str:
    """
    Executa OCR Tesseract em uma região.
    Faz upscale 2× para melhorar acurácia em fontes pequenas.
    """
    upscaled = img_region.resize(
        (img_region.width * 2, img_region.height * 2),
        Image.LANCZOS,
    )
    text = pytesseract.image_to_string(
        upscaled,
        lang=lang,
        config="--psm 6 --oem 3",   # psm 6 = bloco uniforme de texto
    )
    return text.strip()


# ---------------------------------------------------------------------------
# Illustration cleanup (inpainting)
# ---------------------------------------------------------------------------

def _remove_text_from_illustration(
    illustration: Image.Image,
    lang: str = "por",
    inpaint_radius: int = 7,
) -> Image.Image:
    """
    Detecta regiões de texto na ilustração via Tesseract e as remove
    com inpainting OpenCV (TELEA).

    Retorna a ilustração limpa (sem texto sobreposto).
    """
    arr = np.array(illustration.convert("RGB"))

    # --- 1. Detecção de bounding boxes de texto ---
    data = pytesseract.image_to_data(
        illustration,
        lang=lang,
        config="--psm 11 --oem 3",   # psm 11 = texto esparso
        output_type=pytesseract.Output.DICT,
    )

    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    n = len(data["text"])
    for i in range(n):
        conf = int(data["conf"][i])
        text = data["text"][i].strip()
        if conf > 40 and len(text) > 0:
            x, y, w, h = (
                data["left"][i], data["top"][i],
                data["width"][i], data["height"][i],
            )
            # Margem extra ao redor do texto
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(arr.shape[1], x + w + pad)
            y2 = min(arr.shape[0], y + h + pad)
            mask[y1:y2, x1:x2] = 255

    # Se não encontrou texto, retorna original
    text_pixels = mask.sum() / 255
    logger.debug(f"Text pixels found in illustration: {text_pixels}")
    if text_pixels == 0:
        return illustration

    # --- 2. Dilatar a máscara para cobrir bordas de glifos ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # --- 3. Inpainting ---
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.inpaint(arr_bgr, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_pdf(
    pdf_bytes: bytes,
    ocr_lang: str = "por",
    render_scale: float = 2.0,
    clean_illustration: bool = True,
) -> list[PageResult]:
    """
    Pipeline principal de extração.

    Yields events as they happen:
    - {"type": "page_start", "page": N}
    - {"type": "text", "page": N, "text": "..."}
    - {"type": "illustration", "page": N, "image_base64": "..."}
    - {"type": "character", "page": N, "image_base64": "...", "bbox": [...]}
    - {"type": "page_end", "page": N}
    """
    from .vision import character_extractor
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    logger.info(f"Iniciando extração de PDF com {total_pages} páginas (OCR lang: {ocr_lang})")
    
    for page_num in range(total_pages):
        page_idx = page_num + 1
        yield {"type": "page_start", "page": page_idx}
        
        page = doc.load_page(page_num)
        logger.info(f"Processing page {page_idx}/{total_pages}")
        
        # --- 1. Texto (Nativo + OCR Fallback) ---
        native_text = _extract_native_text(page)
        
        # --- Rasterizar página para OCR e Ilustração ---
        composite = _render_page(page, scale=render_scale)
        layout = _detect_layout(composite)
        
        ocr_text = ""
        illus_clean = None
        
        if layout["is_spread"]:
            split_x = layout["split_x"]
            w, h = composite.size
            illus_crop = composite.crop((0, 0, split_x, h))
            text_crop  = composite.crop((split_x, 0, w, h))
            ocr_text = _ocr_region(text_crop, lang=ocr_lang)
            
            if clean_illustration:
                illus_clean = _remove_text_from_illustration(illus_crop, lang=ocr_lang)
            else:
                illus_clean = illus_crop
        else:
            ocr_text = _ocr_region(composite, lang=ocr_lang)
            illus_clean = composite
            
        final_text = ocr_text or native_text
        if final_text:
            yield {"type": "text", "page": page_idx, "text": final_text}
            
        # --- 2. Ilustração ---
        if illus_clean:
            illustration_uri = _to_data_uri(illus_clean)
            yield {"type": "illustration", "page": page_idx, "image_base64": illustration_uri}
            
            # --- 3. Personagens (via vision.py) ---
            try:
                characters = character_extractor.extract_characters(illus_clean)
                for char in characters:
                    char_crop = char["crop"]
                    char_uri = _to_data_uri(char_crop)
                    yield {
                        "type": "character", 
                        "page": page_idx, 
                        "image_base64": char_uri,
                        "bbox": char["bbox"],
                        "method": char["method"],
                        "confidence": char["confidence"]
                    }
            except Exception as e:
                logger.error(f"Erro ao extrair personagens na página {page_idx}: {e}")
                
        yield {"type": "page_end", "page": page_idx}

# ---------------------------------------------------------------------------
# FastAPI endpoint (drop-in)
# ---------------------------------------------------------------------------

def build_fastapi_endpoint():
    """
    Retorna uma função de endpoint FastAPI pronta para usar.
    Cole em seu router assim:

        from pdf_extractor import build_fastapi_endpoint
        router.post("/utils/extract-pdf")(build_fastapi_endpoint())
    """
    from fastapi import UploadFile, File
    from fastapi.responses import JSONResponse
    import dataclasses

    async def extract_pdf_endpoint(file: UploadFile = File(...)):
        """
        Extrai texto (nativo + OCR) e ilustrações de cada página de um PDF.

        Retorna JSON:
        {
          "pages": [
            {
              "page": 1,
              "native_text": "...",   // vazio se PDF for rasterizado
              "ocr_text": "...",      // texto extraído por OCR
              "illustration": "data:image/png;base64,...",
              "raw_images": [
                { "xref": 37, "width": 3444, "height": 2400,
                  "ext": "png", "image_base64": "data:image/png;base64,..." }
              ]
            },
            ...
          ]
        }
        """
        try:
            pdf_bytes = await file.read()
            pages = extract_pdf(pdf_bytes)

            def serialize(obj):
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                raise TypeError

            import json
            payload = json.loads(json.dumps({"pages": pages}, default=serialize))
            return JSONResponse(content=payload)

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": str(e)})

    return extract_pdf_endpoint


# ---------------------------------------------------------------------------
# CLI para teste local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json, dataclasses

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <file.pdf> [output.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"Processing {pdf_path}...")
    pages = extract_pdf(pdf_bytes, clean_illustration=True)

    # Salvar JSON (sem as imagens base64 para legibilidade)
    summary = []
    for p in pages:
        summary.append({
            "page": p.page,
            "native_text": p.native_text,
            "ocr_text": p.ocr_text,
            "illustration_present": p.illustration is not None,
            "raw_images_count": len(p.raw_images),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Summary saved to {out_path}")
    for p in summary:
        print(f"\n--- Page {p['page']} ---")
        print(f"OCR text: {p['ocr_text'][:120]}...")