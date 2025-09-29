import csv
import json
import os
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple


_DATE_RE = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")


def _to_iso(d: str, m: str, y: str) -> Optional[str]:
    try:
        day = int(d)
        mon = int(m)
        yy = int(y)
        if yy < 100:
            yy += 2000 if yy <= 35 else 1900
        return date(yy, mon, day).strftime("%Y-%m-%d")
    except Exception:
        return None


def _to_il(d: str, m: str, y: str) -> Optional[str]:
    try:
        day = int(d)
        mon = int(m)
        yy = int(y)
        if yy < 100:
            yy += 2000 if yy <= 35 else 1900
        return f"{day:02d}/{mon:02d}/{yy:04d}"
    except Exception:
        return None


def _extract_dates(text: str) -> List[Tuple[str, str, str, int]]:
    out: List[Tuple[str, str, str, int]] = []
    for m in _DATE_RE.finditer(text):
        out.append((m.group(1), m.group(2), m.group(3), m.start()))
    return out


def _window(text: str, pos: int, radius: int = 40) -> str:
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end]


KEY_DOCDATE_POS = (
    "תאריך הבדיקה",
    "תאריך הביקור",
    "תאריך שחרור",
    "תאריך הדוח",
    "תאריך מתן התעודה",
    "תאריך ושעה",
    "Date of",
    "Report date",
)
KEY_IGNORE = (
    "תאריך ושעת הדפסת הדוח",
    "תאריך לידה",
    "ת.לידה",
    "לידה",
    "תאריך קבלת הטופס",
)
KEY_CERT_RANGE_START = ("מתאריך", "מועד תחילה")
KEY_CERT_RANGE_END = ("עד תאריך", "תוקף עד", "עד תום")


def _score_date_context(ctx: str) -> int:
    score = 0
    for k in KEY_DOCDATE_POS:
        if k in ctx:
            score += 3
    for k in KEY_IGNORE:
        if k in ctx:
            score -= 5
    if "ביקור" in ctx or "שחרור" in ctx or "בדיקה" in ctx:
        score += 2
    if "הדפסת" in ctx:
        score -= 3
    return score


def _extract_certificate_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    dates = _extract_dates(text)
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    for d, m, y, pos in dates:
        ctx = _window(text, pos)
        iso = _to_iso(d, m, y)
        if not iso:
            continue
        if any(k in ctx for k in KEY_CERT_RANGE_START) and start_iso is None:
            start_iso = iso
        if any(k in ctx for k in KEY_CERT_RANGE_END) and end_iso is None:
            end_iso = iso
    return start_iso, end_iso


def _guess_document_date(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    best_iso: Optional[str] = None
    best_il: Optional[str] = None
    best_score = -10**9
    for d, m, y, pos in _extract_dates(text):
        ctx = _window(text, pos)
        score = _score_date_context(ctx)
        iso = _to_iso(d, m, y)
        ili = _to_il(d, m, y)
        if not iso or not ili:
            continue
        # Prefer recent years
        try:
            yr = int(iso[:4])
            score += (yr - 2000) // 5
        except Exception:
            pass
        if score > best_score:
            best_score = score
            best_iso = iso
            best_il = ili
    sort_date = best_iso
    return best_iso, best_il, sort_date


def _detect_category(text: str, doc_type: Optional[str]) -> Optional[str]:
    t = text or ""
    if doc_type and "תעודה רפואית" in doc_type:
        return "טופס ביטוח"
    if "רפואית תעסוקת" in t or "כושר עבודה" in t or "תעסוקת" in t:
        return "ביקור רופא תעסוקתי"
    if "חדר מיון" in t or "מיון" in t:
        return "ביקור רופא מיון"
    if "פיזיותרפ" in t or "טיפול" in t:
        return "סיכום טיפול"
    return None


def repair_ocr_metadata(ocr_dir: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = ocr_dir
    jsonl_path = os.path.join(ocr_dir, "structured_documents.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    docs: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    fixed: List[Dict[str, Any]] = []
    for doc in docs:
        text = (doc.get("text") or "")
        # Determine document date
        doc_date_iso, doc_date_il, sort_date = _guess_document_date(text)
        # Certificate ranges
        date_start_iso, date_end_iso = (None, None)
        dt = (doc.get("document_type") or "")
        if "תעודה רפואית" in dt or "אישור" in dt:
            s, e = _extract_certificate_range(text)
            date_start_iso, date_end_iso = s, e
            # For certificates, prefer date_end as sort date if present
            if e:
                sort_date = e

        cat = _detect_category(text, doc.get("document_type")) or doc.get("category")

        new_doc = dict(doc)
        if doc_date_iso:
            new_doc["document_date"] = doc_date_iso
            new_doc["document_date_il"] = doc_date_il
        if sort_date:
            new_doc["sort_date"] = sort_date
            try:
                dd = datetime.strptime(sort_date, "%Y-%m-%d").strftime("%d/%m/%Y")
                new_doc["sort_date_il"] = dd
            except Exception:
                pass
        if date_start_iso:
            new_doc["date_start"] = date_start_iso
            try:
                new_doc["date_start_il"] = datetime.strptime(date_start_iso, "%Y-%m-%d").strftime("%d/%m/%Y")
            except Exception:
                pass
        if date_end_iso:
            new_doc["date_end"] = date_end_iso
            try:
                new_doc["date_end_il"] = datetime.strptime(date_end_iso, "%Y-%m-%d").strftime("%d/%m/%Y")
            except Exception:
                pass
        if cat:
            new_doc["category"] = cat
        fixed.append(new_doc)

    # Recompute chronological order id
    def _key(d: Dict[str, Any]) -> Tuple[str, str]:
        return (d.get("sort_date") or d.get("document_date") or "0000-00-00", str(d.get("document_id") or ""))

    fixed.sort(key=_key)
    for idx, d in enumerate(fixed):
        d["chron_id"] = int(idx + 1)

    # Write fixed JSONL
    fixed_jsonl = os.path.join(out_dir, "structured_documents.jsonl")
    with open(fixed_jsonl, "w", encoding="utf-8") as f:
        for d in fixed:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Write documents_index.json
    index_map: Dict[str, Any] = {}
    for d in fixed:
        did = str(d.get("document_id") or d.get("original_id") or "")
        if not did:
            continue
        index_map[did] = {
            "original_id": d.get("original_id"),
            "chron_id": d.get("chron_id"),
            "document_type": d.get("document_type"),
            "document_date": d.get("document_date"),
            "issuer": d.get("issuer"),
            "pages": d.get("pages"),
            "source": d.get("source"),
            "patient_name": d.get("patient_name"),
        }
    idx_path = os.path.join(out_dir, "documents_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)

    # Write CSV
    csv_path = os.path.join(out_dir, "clean_documents.csv")
    fields = [
        "chron_id",
        "document_id",
        "original_id",
        "document_type",
        "category",
        "issuer",
        "document_date",
        "document_date_il",
        "date_start",
        "date_end",
        "sort_date",
        "pages",
        "patient_name",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in fixed:
            row = {k: d.get(k) for k in fields}
            w.writerow(row)

    return {
        "jsonl": fixed_jsonl,
        "index": idx_path,
        "csv": csv_path,
        "num_docs": len(fixed),
    }


__all__ = ["repair_ocr_metadata"]


