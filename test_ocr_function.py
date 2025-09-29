#!/usr/bin/env python3
"""
Test script to verify OCR function is working.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)

try:
    from src.ocr_pipeline import ocr_pdf_best
    print("✓ OCR pipeline imported successfully")

    # Test with a small PDF if available
    test_pdf = "med_patient#1.pdf"
    if os.path.exists(test_pdf):
        print(f"Testing OCR on {test_pdf}...")
        try:
            result = ocr_pdf_best(
                pdf_path=test_pdf,
                output_dir="test_ocr_output",
                dpi=150,  # Lower DPI for faster test
                max_pages=1,  # Just first page
                bidi_mode="visual"
            )
            print("✓ OCR completed successfully")
            print(f"Result keys: {list(result.keys())}")
            if 'pages' in result and result['pages']:
                first_page = result['pages'][0]
                print(f"First page text preview: {first_page.get('text', '')[:200]}...")
        except Exception as e:
            print(f"✗ OCR failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Test PDF {test_pdf} not found")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()