import os
import sys
import json
from typing import Optional

import click
from rich import print
from rich.console import Console
from rich.table import Table

try:
    from .downloader import download_synthea_fhir_r4  # type: ignore
except Exception:  # pragma: no cover - optional module
    download_synthea_fhir_r4 = None  # type: ignore
try:
    from .fhir_parse import index_fhir_directory, extract_medications_for_patient  # type: ignore
except Exception:  # pragma: no cover - optional module
    index_fhir_directory = None  # type: ignore
    extract_medications_for_patient = None  # type: ignore
try:
    from .query import parse_patient_query  # type: ignore
except Exception:  # pragma: no cover - optional module
    parse_patient_query = None  # type: ignore
try:
    from .summarize import summarize_medications  # type: ignore
except Exception:  # pragma: no cover - optional module
    summarize_medications = None  # type: ignore
try:
    from .rag import build_embeddings, rag_query_medications  # type: ignore
except Exception:  # pragma: no cover - optional module
    build_embeddings = None  # type: ignore
    rag_query_medications = None  # type: ignore
from .rag_ocr import index_ocr_corpus, search_ocr_index
try:
    from .llm_ollama import ollama_stream  # type: ignore
except Exception:  # pragma: no cover - optional module
    ollama_stream = None  # type: ignore
try:
    from .rag_answer import rag_answer_json  # type: ignore
except Exception:  # pragma: no cover - optional module
    rag_answer_json = None  # type: ignore
try:
    from .case_report import build_case_report  # type: ignore
except Exception:  # pragma: no cover - optional module
    build_case_report = None  # type: ignore
try:
    from .ragas_eval import run_ragas  # type: ignore
except Exception:  # pragma: no cover - optional module
    run_ragas = None  # type: ignore
try:
    from .ocr_ragas_eval import build_hebrew_ragas_samples, save_jsonl  # type: ignore
except Exception:  # pragma: no cover - optional module
    build_hebrew_ragas_samples = None  # type: ignore
    save_jsonl = None  # type: ignore
try:
    from .ragas_sweep import sweep_k  # type: ignore
except Exception:  # pragma: no cover - optional module
    sweep_k = None  # type: ignore
try:
    from .ocr_fix import repair_ocr_metadata  # type: ignore
except Exception:  # pragma: no cover - optional module
    repair_ocr_metadata = None  # type: ignore
from .ocr_audit import audit_results
from .results_sync import sync_aligned_files
try:
    from .csv_sync import csv_to_structured_sync  # type: ignore
except Exception:  # pragma: no cover - optional module
    csv_to_structured_sync = None  # type: ignore


console = Console()


def default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--data-dir", type=click.Path(file_okay=False), default=default_data_dir())
@click.option("--url", type=str, default=None)
def download(data_dir: str, url: Optional[str]) -> None:
    if download_synthea_fhir_r4 is None:
        console.print("download command is unavailable (missing optional module)")
        sys.exit(1)
    out = download_synthea_fhir_r4(data_dir, url=url)  # type: ignore
    print(out)


@cli.command()
@click.option("--data-dir", type=click.Path(file_okay=False), default=default_data_dir())
def index(data_dir: str) -> None:
    root = os.path.join(data_dir, "synthea_fhir_r4")
    if index_fhir_directory is None:
        console.print("index command is unavailable (missing optional module)")
        sys.exit(1)
    idx = index_fhir_directory(root)  # type: ignore
    out_path = os.path.join(data_dir, "index.json")
    with open(out_path, "w") as f:
        json.dump(idx, f)
    print(out_path)


@cli.command()
@click.option("--data-dir", type=click.Path(file_okay=False), default=default_data_dir())
def list_patients(data_dir: str) -> None:
    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        console.print("Index not found. Run: python -m src.cli index")
        sys.exit(1)
    data = json.load(open(index_path))
    table = Table(title="Indexed Patients")
    table.add_column("Patient ID")
    table.add_column("Resources")
    for pid, bucket in data.items():
        table.add_row(pid, ", ".join(sorted(bucket.keys())))
    console.print(table)


@cli.command()
@click.argument("query_text", nargs=-1)
@click.option("--data-dir", type=click.Path(file_okay=False), default=default_data_dir())
def ask(query_text: tuple, data_dir: str) -> None:
    text = " ".join(query_text)
    if parse_patient_query is None:
        console.print("ask command is unavailable (missing optional module)")
        sys.exit(1)
    patient_id, info_kind = parse_patient_query(text)  # type: ignore
    if not patient_id or info_kind is None:
        print("Could not parse patient id or info type from query.")
        sys.exit(1)

    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        console.print("Index not found. Run: python -m src.cli index")
        sys.exit(1)
    idx = json.load(open(index_path))

    if info_kind == "medications":
        if extract_medications_for_patient is None or summarize_medications is None:
            console.print("ask medications is unavailable (missing optional modules)")
            sys.exit(1)
        meds = extract_medications_for_patient(idx, patient_id)  # type: ignore
        summary = summarize_medications(patient_id, meds)  # type: ignore
        print(summary)
    else:
        print(f"Info type '{info_kind}' not yet implemented.")


@cli.command()
@click.option("--data-dir", type=click.Path(file_okay=False), default=default_data_dir())
def embed(data_dir: str) -> None:
    if build_embeddings is None:
        console.print("embed command is unavailable (missing optional module)")
        sys.exit(1)
    index_path, meta_path = build_embeddings(data_dir)  # type: ignore
    print(index_path)
    print(meta_path)


@cli.command()
@click.argument("patient_id")
@click.argument("query_text", nargs=-1)
@click.option("--data-dir", type=click.Path(file_okay=False), default=default_data_dir())
@click.option("--top-k", type=int, default=8)
def rag(patient_id: str, query_text: tuple, data_dir: str, top_k: int) -> None:
    text = " ".join(query_text)
    if rag_query_medications is None:
        console.print("rag command is unavailable (missing optional module)")
        sys.exit(1)
    docs = rag_query_medications(data_dir, patient_id, text, top_k=top_k)  # type: ignore
    if not docs:
        print("No relevant medication entries found.")
        return
    for d in docs:
        print(d.get("text"))


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--chunk-chars", type=int, default=900)
@click.option("--overlap", type=int, default=120)
@click.option("--model", "model_name", type=str, default=None, help="SentenceTransformer model name (default: env HE_EMBED_MODEL or intfloat/multilingual-e5-large)")
@click.option("--multi/--no-multi", "enable_multi", default=True, help="Enable multi-vector indexing (title/keywords/summary/body)")
def ocr_index(ocr_dir: str, chunk_chars: int, overlap: int, model_name: str | None, enable_multi: bool) -> None:
    idx, meta = index_ocr_corpus(ocr_dir, chunk_chars=chunk_chars, overlap_chars=overlap, model_name=model_name, enable_multi_vector=enable_multi)
    print(idx)
    print(meta)


@cli.command()
@click.argument("query_text", nargs=-1)
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--top-k", type=int, default=8)
@click.option("--type", "doc_type", type=str, default=None, help="Optional filter: document_type in Hebrew")
@click.option("--category", type=str, default=None, help="Optional filter: normalized category (ביקור רופא כללי/תעסוקתי/מיון/טופס ביטוח/אחר)")
def ocr_search(query_text: tuple, ocr_dir: str, top_k: int, doc_type: str | None, category: str | None) -> None:
    text = " ".join(query_text)
    filters = {}
    if doc_type:
        filters["document_type"] = doc_type
    if category:
        filters["category"] = category
    if not filters:
        filters = None
    rows = search_ocr_index(ocr_dir, text, top_k=top_k, filters=filters)
    if not rows:
        print("No results.")
        return
    for r in rows:
        print(json.dumps({
            "row_id": r.get("row_id"),
            "document_type": r.get("document_type"),
            "category": r.get("category"),
            "document_date": r.get("document_date"),
            "pages": r.get("pages"),
            "quote": (r.get("text") or "")[:220],
            "matched_field": r.get("__matched_field__"),
        }, ensure_ascii=False))


@cli.command()
@click.argument("query_text", nargs=-1)
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--top-k", type=int, default=6)
@click.option("--category", type=str, default=None)
@click.option("--type", "doc_type", type=str, default=None)
@click.option("--model", type=str, default="gemma2:2b-instruct")
def ocr_answer(query_text: tuple, ocr_dir: str, top_k: int, category: str | None, doc_type: str | None, model: str) -> None:
    text = " ".join(query_text)
    out = rag_answer_json(ocr_dir, text, top_k=top_k, model=model, category=category, doc_type=doc_type)
    print(json.dumps(out, ensure_ascii=False))


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--model", type=str, default="qwen2.5:7b-instruct")
@click.option("--top-k", type=int, default=6)
def ocr_case_report(ocr_dir: str, model: str, top_k: int) -> None:
    """Generate multi-question answers and a chronological summary from OCR corpus."""
    if build_case_report is None:
        console.print("case_report is unavailable (missing optional module)")
        sys.exit(1)
    report = build_case_report(ocr_dir, model=model, top_k=top_k)  # type: ignore
    print(json.dumps(report, ensure_ascii=False, indent=2))


@cli.command()
@click.argument("query_text", nargs=-1)
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--top-k", type=int, default=6)
@click.option("--category", type=str, default=None)
@click.option("--type", "doc_type", type=str, default=None)
@click.option("--model", type=str, default="qwen2.5:7b-instruct")
def ocr_stream(query_text: tuple, ocr_dir: str, top_k: int, category: str | None, doc_type: str | None, model: str) -> None:
    """Stream a concise Hebrew answer with retrieval context."""
    text = " ".join(query_text)
    filters = {}
    if category:
        filters["category"] = category
    if doc_type:
        filters["document_type"] = doc_type
    if not filters:
        filters = None
    rows = search_ocr_index(ocr_dir, text, top_k=top_k, filters=filters)
    if not rows:
        print("לא נמצאו מקורות.")
        return
    from .rag_answer import build_context_rows, SYSTEM_PROMPT_HE, prompt_for_question
    context = build_context_rows(rows)
    prompt = prompt_for_question(text, context)
    if ollama_stream is None:
        console.print("ollama streaming is unavailable (missing optional module)")
        sys.exit(1)
    for chunk in ollama_stream(prompt, model=model, system=SYSTEM_PROMPT_HE, temperature=0.2, as_json=True):  # type: ignore
        print(chunk, end="", flush=True)
    print()


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--model", type=str, default="qwen2.5:7b-instruct")
@click.option("--top-k", type=int, default=6)
def case_report_export(ocr_dir: str, model: str, top_k: int) -> None:
    """Export answers + chronology in the requested tabular-like JSON format."""
    if build_case_report is None:
        console.print("case_report_export is unavailable (missing optional module)")
        sys.exit(1)
    report = build_case_report(ocr_dir, model=model, top_k=top_k)  # type: ignore
    # Reformat for table-like output
    out = {
        "answers_table": [
            {
                "question": a.get("question"),
                "answer": a.get("answer"),
                "sources": a.get("sources"),
            }
            for a in report.get("answers", [])
        ],
        "chronology_table": report.get("chronology", []),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--input", "jsonl_path", type=click.Path(dir_okay=False, exists=True), required=True)
def ragas_eval(jsonl_path: str) -> None:
    """Run RAGAS evaluation on a JSONL file with fields: question, answer, ground_truths, contexts."""
    if run_ragas is None:
        console.print("ragas_eval is unavailable (missing optional module)")
        sys.exit(1)
    scores = run_ragas(jsonl_path)  # type: ignore
    print(json.dumps(scores, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--model", type=str, default="qwen2.5:7b-instruct")
@click.option("--top-k", type=int, default=6)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out", "ragas_he.jsonl")))
def ocr_build_ragas(ocr_dir: str, model: str, top_k: int, out_path: str) -> None:
    """Build a Hebrew RAGAS dataset from the OCR corpus for the 3 core questions."""
    samples = build_hebrew_ragas_samples(ocr_dir, model=model, top_k=top_k)
    p = save_jsonl(samples, out_path)
    print(p)


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--out-dir", type=click.Path(file_okay=False), default=None)
def ocr_fix(ocr_dir: str, out_dir: str | None) -> None:
    """Repair dates/categories/chronology in existing OCR structured data without re-extraction.

    - Scans each document text for realistic document_date and ignores print dates/birthdays
    - Detects sick-leave ranges (date_start/date_end) where applicable
    - Rewrites structured_documents.jsonl and documents_index.json
    - Exports a CSV summary for human review
    """
    out = repair_ocr_metadata(ocr_dir, out_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
def ocr_audit(ocr_dir: str) -> None:
    """Audit and reflow OCR text per page using existing results.json; write results_aligned.json, full_text_aligned.txt, and audit_pages.csv."""
    out = audit_results(ocr_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--reindex/--no-reindex", default=True, help="Rebuild OCR embeddings after syncing")
@click.option("--model", "model_name", type=str, default=None)
@click.option("--chunk-chars", type=int, default=900)
@click.option("--overlap", type=int, default=120)
def ocr_sync(ocr_dir: str, reindex: bool, model_name: str | None, chunk_chars: int, overlap: int) -> None:
    """Synchronize results_aligned.json and results_aligned_cleaned.json to be identical; regenerate full_text_aligned.txt; rebuild structured docs; optionally rebuild embeddings."""
    # Local import to avoid optional dependency errors at module import time
    from .ocr_structuring import build_ocr_documents  # type: ignore
    sync_out = sync_aligned_files(ocr_dir)
    # Rebuild structured docs from canonical JSON
    aligned = os.path.join(ocr_dir, "results_aligned.json")
    try:
        jsonl_path, index_path = build_ocr_documents(aligned if os.path.exists(aligned) else os.path.join(ocr_dir, "results.json"), ocr_dir)
    except Exception as e:
        print(json.dumps({"error": f"structuring failed: {e}"}, ensure_ascii=False))
        jsonl_path = os.path.join(ocr_dir, "structured_documents.jsonl")
        index_path = os.path.join(ocr_dir, "documents_index.json")

    result = {"sync": sync_out, "structured": {"jsonl": jsonl_path, "index": index_path}}

    if reindex:
        try:
            idx, meta = index_ocr_corpus(ocr_dir, chunk_chars=chunk_chars, overlap_chars=overlap, model_name=model_name)
            result["embeddings"] = {"index": idx, "meta": meta}
        except Exception as e:
            result["embeddings_error"] = str(e)

    print(json.dumps(result, ensure_ascii=False, indent=2))

@cli.command()
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False, exists=True), required=True)
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
def csv_sync(csv_path: str, ocr_dir: str) -> None:
    """Synchronize structured_documents.jsonl and documents_index.json to match the corrected CSV exactly."""
    jsonl, idx = csv_to_structured_sync(csv_path, ocr_dir)
    print(json.dumps({"jsonl": jsonl, "index": idx}, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--ocr-dir", type=click.Path(file_okay=False), default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ocr_out")))
@click.option("--model", type=str, default="qwen2.5:7b-instruct")
def ragas_sweep(ocr_dir: str, model: str) -> None:
    """Sweep k values and output best by faithfulness then answer_relevancy."""
    out = sweep_k(ocr_dir, ks=(6, 8, 10, 12, 14), model=model, base_answer_k=10)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()


