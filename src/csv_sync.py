import csv
import json
import os
import ast
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


def _read_results_pages(ocr_dir: str) -> List[Dict[str, Any]]:
    aligned = os.path.join(ocr_dir, "results_aligned.json")
    base = os.path.join(ocr_dir, "results.json")
    path = aligned if os.path.isfile(aligned) else base
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pages") or []


def _parse_pages_field(pages_field: str) -> List[int]:
    if not pages_field:
        return []
    try:
        # Try JSON
        val = json.loads(pages_field)
        if isinstance(val, list):
            return [int(x) for x in val]
    except Exception:
        pass
    try:
        # Try Python literal (e.g. "[26, 27]")
        val = ast.literal_eval(pages_field)
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
    except Exception:
        pass
    # Fallback: comma-separated
    out: List[int] = []
    for tok in pages_field.replace("[", "").replace("]", "").split(","):
        tok = tok.strip()
        if tok:
            try:
                out.append(int(tok))
            except Exception:
                continue
    return out


def _to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    # Try ISO first
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _to_il_date(iso: Optional[str]) -> Optional[str]:
    if not iso:
        return None
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return None


def _combine_pages_text(ocr_dir: str, pages: List[int]) -> str:
    pages_data = _read_results_pages(ocr_dir)
    page_map = {int(p.get("page") or 0): (p.get("text") or "") for p in pages_data}
    chunks: List[str] = []
    for pn in sorted(set(int(x) for x in pages)):
        txt = page_map.get(int(pn)) or ""
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)


def csv_to_structured_sync(csv_path: str, ocr_dir: str) -> Tuple[str, str]:
    # Read rows, normalize header (first column may be unnamed chron_id)
    with open(csv_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Ensure no stray BOM
    raw = raw.lstrip("\ufeff")
    lines = raw.splitlines()
    if not lines:
        raise RuntimeError("CSV is empty")
    header = [h.strip() for h in lines[0].split(",")]
    if header and (header[0] == "" or header[0] == ","):
        header[0] = "chron_id"
        lines[0] = ",".join(header)
    # Parse CSV with corrected header
    reader = csv.DictReader(lines)
    rows = list(reader)

    # Build structured_documents.jsonl
    jsonl_path = os.path.join(ocr_dir, "structured_documents.jsonl")
    index_path = os.path.join(ocr_dir, "documents_index.json")

    docs: List[Dict[str, Any]] = []
    index: Dict[str, Any] = {}
    for r in rows:
        chron_id = int((r.get("chron_id") or r.get("Chron") or "0") or 0)
        document_id = (r.get("document_id") or r.get("doc_id") or str(chron_id)).strip()
        document_type = (r.get("document_type") or r.get("type") or "").strip()
        category = (r.get("category") or "").strip()
        issuer = (r.get("issuer") or "").strip()
        patient_name = (r.get("patient_name") or "").strip()
        pages_field = r.get("pages") or ""
        pages = _parse_pages_field(pages_field)

        # Dates
        doc_date_iso = _to_iso_date(r.get("document_date"))
        date_start_iso = _to_iso_date(r.get("date_start"))
        date_end_iso = _to_iso_date(r.get("date_end"))
        sort_date = doc_date_iso or date_end_iso or date_start_iso

        # Text from pages
        full_text = _combine_pages_text(ocr_dir, pages)

        doc: Dict[str, Any] = {
            "document_id": document_id,
            "original_id": document_id,
            "document_type": document_type,
            "document_date": doc_date_iso,
            "issuer": issuer or None,
            "pages": pages,
            "text": full_text,
            "source": None,
            "patient_name": patient_name or None,
            "found_dates": [d for d in [doc_date_iso, date_start_iso, date_end_iso] if d],
            "detected_keywords": [],
            "category": category or "אחר",
            "date_start": date_start_iso,
            "date_end": date_end_iso,
            "sort_date": sort_date,
            "document_date_il": _to_il_date(doc_date_iso),
            "date_start_il": _to_il_date(date_start_iso),
            "date_end_il": _to_il_date(date_end_iso),
            "sort_date_il": _to_il_date(sort_date),
            "chron_id": int(chron_id),
        }
        docs.append(doc)

        index[document_id] = {
            "original_id": document_id,
            "chron_id": int(chron_id),
            "document_type": document_type,
            "document_date": doc_date_iso,
            "issuer": issuer or None,
            "pages": pages,
            "source": None,
            "patient_name": patient_name or None,
        }

    # Sort docs by chron_id ascending (oldest to newest), then write newest-first like build_ocr_documents
    docs_sorted_asc = sorted(docs, key=lambda d: int(d.get("chron_id") or 0))
    docs_sorted_desc = list(reversed(docs_sorted_asc))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs_sorted_desc:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return jsonl_path, index_path


__all__ = ["csv_to_structured_sync"]


def fix_chron_by_doc_date(ocr_dir: str) -> Tuple[str, str]:
    """Recompute chron_id strictly by document_date ascending and update both JSONL and index.

    - If document_date is missing, fallback to sort_date, then date_end, then date_start; if still missing, place last.
    - Writes back to structured_documents.jsonl (preserving current row order), and updates documents_index.json.
    """
    jsonl_path = os.path.join(ocr_dir, "structured_documents.jsonl")
    index_path = os.path.join(ocr_dir, "documents_index.json")
    if not os.path.isfile(jsonl_path) or not os.path.isfile(index_path):
        raise FileNotFoundError("structured_documents.jsonl or documents_index.json not found")

    docs: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    def key_fn(d: Dict[str, Any]) -> Tuple[str, str]:
        for k in ("document_date", "sort_date", "date_end", "date_start"):
            v = d.get(k)
            if isinstance(v, str) and v:
                return (v, str(d.get("document_id") or ""))
        return ("9999-99-99", str(d.get("document_id") or ""))

    docs_sorted = sorted(docs, key=key_fn)
    id_to_chron: Dict[str, int] = {}
    for idx, d in enumerate(docs_sorted, start=1):
        did = str(d.get("document_id") or "")
        if did and did not in id_to_chron:
            id_to_chron[did] = idx

    # Update docs in-place preserving current file order
    for d in docs:
        did = str(d.get("document_id") or "")
        if did in id_to_chron:
            d["chron_id"] = int(id_to_chron[did])

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Update index
    with open(index_path, "r", encoding="utf-8") as f:
        idx_map = json.load(f)
    for did, entry in idx_map.items():
        if did in id_to_chron:
            entry["chron_id"] = int(id_to_chron[did])
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(idx_map, f, ensure_ascii=False, indent=2)

    return jsonl_path, index_path


__all__.append("fix_chron_by_doc_date")


