import os
import sys
import click
from rich import print
from .ocr_pipeline import ocr_pdf_best
from .ocr_structuring import build_ocr_documents
from .ocr_audit import audit_results


@click.command()
@click.argument("pdf_path")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="ocr_out")
@click.option("--dpi", type=int, default=300)
@click.option("--highres-dpi", type=int, default=600)
@click.option("--prefer-vector-text/--no-prefer-vector-text", default=False, help="Use PDF embedded text layer when available")
@click.option("--bidi", type=click.Choice(["visual", "logical"]), default="visual", help="Text order for RTL scripts: visual (display) or logical (storage)")
@click.option("--max-pages", type=int, default=None, help="Limit number of pages to process")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto")
@click.option("--preprocess/--no-preprocess", default=True, help="Enable adaptive preprocessing (deskew, dewarp, shadow removal, etc.)")
def main(pdf_path: str, out_dir: str, dpi: int, highres_dpi: int, prefer_vector_text: bool, bidi: str, max_pages: int | None, device: str, preprocess: bool) -> None:
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    dev = None if device == "auto" else device
    summary = ocr_pdf_best(
        pdf_path,
        out_dir,
        dpi=dpi,
        highres_dpi=highres_dpi,
        prefer_vector_text=prefer_vector_text,
        max_pages=max_pages,
        device=dev,
        bidi_mode=bidi,
        preprocess=preprocess,
    )
    print(f"Wrote OCR to {out_dir}")
    print(f"Pages: {summary['num_pages']}")
    # Audit/reflow per page without re-extraction, then build structured docs
    results_json = os.path.join(out_dir, "results.json")
    try:
        aud = audit_results(out_dir)
        aligned_results = os.path.join(out_dir, "results_aligned.json")
        jsonl_path, index_path = build_ocr_documents(aligned_results if os.path.exists(aligned_results) else results_json, out_dir)
        print(f"Structured documents: {jsonl_path}")
        print(f"Documents index: {index_path}")
        print(f"Audit written: {aud['audit_csv']}")
    except Exception as e:
        print(f"[yellow]Warning:[/yellow] could not build structured documents: {e}")


if __name__ == "__main__":
    main()



