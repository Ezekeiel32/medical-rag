import io
import json
import os
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from bidi.algorithm import get_display
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None
try:
    # Optional preprocessing for OCR images
    from .ocr_preprocess import preprocess_image, PreprocessConfig
except Exception:
    preprocess_image = None
    PreprocessConfig = None


@dataclass
class OcrPageResult:
    page_index: int
    width: int
    height: int
    text: str
    lines: List[Dict[str, Any]]


def _render_pdf_page(doc: fitz.Document, page_index: int, dpi: int = 300) -> Image.Image:
    page = doc.load_page(page_index)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _extract_text_if_vector_pdf(doc: fitz.Document) -> List[str]:
    # Extract selectable text as a fast path; if page has no text, return empty string for that page
    texts: List[str] = []
    for page in doc:
        txt = page.get_text("text") or ""
        # Normalize unicode; do not apply bidi here (logical order likely correct for text layer)
        texts.append(unicodedata.normalize("NFC", txt))
    return texts

def _extract_text_by_words(page: fitz.Page) -> str:
    """
    Reconstruct per-line text from words to avoid encoding/PUA artifacts.
    Uses PyMuPDF 'words' with (x0,y0,x1,y1,word,block_no,line_no,word_no).
    """
    try:
        words = page.get_text("words") or []
    except Exception:
        words = []
    if not words:
        # Fallback to default text if words API unavailable
        return page.get_text("text") or ""

    # Group by (block_no, line_no)
    lines_map: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}
    for w in words:
        if not isinstance(w, (list, tuple)) or len(w) < 8:
            # Unexpected shape; fallback
            return page.get_text("text") or ""
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        if not text:
            continue
        key = (int(block_no), int(line_no))
        lines_map.setdefault(key, []).append((float(x0), str(text)))

    # Sort lines by (block_no, line_no) and words by x0
    out_lines: List[str] = []
    for key in sorted(lines_map.keys()):
        parts = [t for _, t in sorted(lines_map[key], key=lambda t: t[0])]
        # Join with spaces and normalize
        line = " ".join(parts)
        line = unicodedata.normalize("NFC", line)
        out_lines.append(line)

    # Join lines with newlines
    return "\n".join(out_lines).strip()


def _extract_lines_with_bboxes(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extract lines with their bounding boxes using PyMuPDF 'dict' structure.
    Each line: { "text": str (NFC), "bbox": [x0,y0,x1,y1] }
    Coordinates are in PDF space (points).
    """
    try:
        d = page.get_text("dict")
    except Exception:
        d = None
    if not d:
        return []

    out: List[Dict[str, Any]] = []
    for block in d.get("blocks", []):
        # Only text blocks
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []) or []:
            bbox = line.get("bbox") or None
            spans = line.get("spans", []) or []
            parts: List[str] = []
            for sp in spans:
                t = sp.get("text") or ""
                if t:
                    parts.append(t)
            txt = " ".join(parts).strip()
            if not txt:
                continue
            try:
                b = [float(v) for v in (bbox or [])] if bbox else None
            except Exception:
                b = None
            out.append({"text": unicodedata.normalize("NFC", txt), "bbox": b})
    return out

def _char_stats(text: str) -> Dict[str, float]:
    t = text or ""
    n = len(t)
    if n == 0:
        return {"n": 0, "he_ratio": 0.0, "ipa_ratio": 0.0, "pua_ratio": 0.0}
    he = sum(1 for ch in t if "\u0590" <= ch <= "\u05FF")
    ipa = sum(1 for ch in t if "\u02B0" <= ch <= "\u02FF")
    pua = sum(1 for ch in t if "\uE000" <= ch <= "\uF8FF")
    return {
        "n": float(n),
        "he_ratio": he / n,
        "ipa_ratio": ipa / n,
        "pua_ratio": pua / n,
    }

def _is_text_corrupted(text: str) -> bool:
    if not text or not text.strip():
        return True
    s = _char_stats(text)
    # Heuristic: many IPA/PUA glyphs or very low Hebrew presence
    if s["ipa_ratio"] > 0.05 or s["pua_ratio"] > 0.02:
        return True
    if s["he_ratio"] < 0.03 and (s["ipa_ratio"] > 0.0 or s["pua_ratio"] > 0.0):
        return True
    return False

def _extract_text_pdfminer(pdf_path: str, page_index: int) -> str:
    """
    Extract text from a single page using pdfminer.six as a fallback.
    page_index is 0-based.
    """
    if pdfminer_extract_text is None:
        return ""
    try:
        txt = pdfminer_extract_text(pdf_path, page_numbers=[page_index]) or ""
        return unicodedata.normalize("NFC", txt)
    except Exception:
        return ""

def _apply_bidi_per_line(text: str, mode: str = "visual") -> str:
    if mode != "visual":
        return text
    return "\n".join(get_display(line) for line in (text or "").splitlines())


def ocr_pdf_best(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    highres_dpi: int = 600,
    prefer_vector_text: bool = False,  # Always use Surya OCR
    max_pages: Optional[int] = None,
    device: Optional[str] = None,
    bidi_mode: str = "visual",
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    OCR PDF using Surya OCR with proper Hebrew support.
    Replaces the bogus vector text extraction with real OCR.
    """
    # Import and use the Surya OCR implementation
    from .surya_ocr_pipeline import surya_ocr_pdf
    
    return surya_ocr_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=dpi,
        highres_dpi=highres_dpi,
        prefer_vector_text=prefer_vector_text,
        max_pages=max_pages,
        device=device,
        bidi_mode=bidi_mode,
        preprocess=preprocess,
    )
