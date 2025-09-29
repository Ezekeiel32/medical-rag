"""
Surya OCR Pipeline for Hebrew Medical Documents
Replaces the bogus ocr_pdf_best with proper Surya OCR implementation.
"""

import os
import json
import fitz
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional
from bidi.algorithm import get_display

try:
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    from surya.languages import CODE_TO_LANGUAGE
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False
    print("WARNING: Surya OCR not available. Install with: pip install surya-ocr")


def _render_pdf_page(doc: fitz.Document, page_index: int, dpi: int = 300) -> Image.Image:
    """Render PDF page to PIL Image."""
    page = doc.load_page(page_index)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _apply_bidi_processing(text: str, bidi_mode: str = "logical") -> str:
    """Apply bidirectional text processing for Hebrew."""
    if bidi_mode == "visual":
        # Apply bidi per line to avoid scrambling mixed-direction text
        return "\n".join(get_display(line) for line in (text or "").splitlines())
    return text


def surya_ocr_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    highres_dpi: int = 600,
    prefer_vector_text: bool = False,  # Always use OCR, not vector text
    max_pages: Optional[int] = None,
    device: Optional[str] = None,
    bidi_mode: str = "visual",
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    Extract text from PDF using Surya OCR with proper Hebrew support.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save results
        dpi: DPI for text detection (first pass)
        highres_dpi: DPI for text recognition (second pass)
        prefer_vector_text: Ignored - always use OCR
        max_pages: Maximum pages to process
        device: Device to use ('cuda' or 'cpu')
        bidi_mode: BiDi processing mode ('visual' or 'logical')
        preprocess: Whether to preprocess images
    
    Returns:
        Dictionary with OCR results
    """
    if not SURYA_AVAILABLE:
        raise RuntimeError("Surya OCR not available. Install with: pip install surya-ocr")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Surya models
    print("Loading Surya OCR models...")
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    
    # Open PDF
    doc = fitz.open(pdf_path)
    try:
        num_pages = doc.page_count
        limit = min(num_pages, max_pages) if max_pages else num_pages
        
        print(f"Processing {limit} pages with Surya OCR...")
        
        # Process pages
        structured_pages: List[Dict[str, Any]] = []
        combined_text_pages: List[str] = []
        
        for i in range(limit):
            print(f"Processing page {i + 1}/{limit}...")
            
            # Render page to image
            img = _render_pdf_page(doc, i, dpi=dpi)
            
            # Convert to numpy array for Surya
            img_array = np.array(img)
            
            # Run Surya OCR
            try:
                # Convert numpy array back to PIL Image
                img_pil = Image.fromarray(img_array)
                
                # Run Surya OCR
                det_result = run_ocr(
                    images=[img_pil],
                    langs=[["he", "en"]],  # Hebrew and English
                    det_model=det_model,
                    det_processor=det_processor,
                    rec_model=rec_model,
                    rec_processor=rec_processor
                )
                
                # Extract text and bounding boxes
                page_result = det_result[0]
                lines = []
                page_text_lines = []
                
                for line in page_result.text_lines:
                    # Extract text
                    line_text = line.text
                    if line_text.strip():
                        # Apply BiDi processing
                        processed_text = _apply_bidi_processing(line_text, bidi_mode)
                        
                        # Get bounding box
                        bbox = line.bbox
                        
                        lines.append({
                            "text": processed_text,
                            "bbox": bbox
                        })
                        page_text_lines.append(processed_text)
                
                # Combine all lines into page text
                page_text = "\n".join(page_text_lines)
                
                # Get page dimensions
                w, h = img.size
                
                structured_pages.append({
                    "page": i + 1,
                    "width": w,
                    "height": h,
                    "text": page_text,
                    "lines": lines,
                })
                combined_text_pages.append(page_text)
                
            except Exception as e:
                print(f"Error processing page {i + 1}: {e}")
                # Add empty page on error
                structured_pages.append({
                    "page": i + 1,
                    "width": 0,
                    "height": 0,
                    "text": "",
                    "lines": [],
                })
                combined_text_pages.append("")
        
        # Create results summary
        summary = {
            "source": os.path.abspath(pdf_path),
            "num_pages": len(structured_pages),
            "pages": structured_pages,
        }
        
        # Save results
        with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(output_dir, "full_text.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_text_pages))
        
        print(f"Surya OCR completed. Processed {len(structured_pages)} pages.")
        return summary
        
    finally:
        doc.close()


def ocr_pdf_best(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    highres_dpi: int = 600,
    prefer_vector_text: bool = False,
    max_pages: Optional[int] = None,
    device: Optional[str] = None,
    bidi_mode: str = "visual",
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    Main OCR function - now uses Surya OCR instead of bogus vector text extraction.
    """
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
