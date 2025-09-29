import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

from .ocr_structuring import build_ocr_documents


HE_EMBED_MODEL = os.environ.get("HE_EMBED_MODEL", "intfloat/multilingual-e5-large")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_model(model_name: Optional[str] = None) -> SentenceTransformer:
    name = model_name or HE_EMBED_MODEL
    try:
        return SentenceTransformer(name, device=_device())
    except Exception:
        return SentenceTransformer(name, device="cpu")


def _index_path(embed_dir: str) -> str:
    return os.path.join(embed_dir, "index.faiss")


def _meta_path(embed_dir: str) -> str:
    return os.path.join(embed_dir, "meta.jsonl")


def _vectors_path(embed_dir: str) -> str:
    return os.path.join(embed_dir, "vectors.npy")


def _model_info_path(embed_dir: str) -> str:
    return os.path.join(embed_dir, "model.json")


def _split_sentences_he(text: str) -> List[str]:
    # Simple sentence splitter for Hebrew/RTL; keep punctuation.
    parts = re.split(r"(?<=[.!?\u05BE\u05C3])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


_CTRL_RE = re.compile(r"[\x00-\x1F\x7F]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _denoise_text(text: str) -> str:
    # Strip control characters and simple HTML remnants, collapse spaces
    t = _CTRL_RE.sub(" ", text or "")
    t = _HTML_TAG_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t.strip()


def _chunk_text(text: str, target_chars: int = 900, overlap_chars: int = 120) -> List[str]:
    """
    Create raw character windows preserving the text EXACTLY as in source.
    No normalization, no trimming, no sentence reflow.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + target_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        # Advance with overlap
        start = end - overlap_chars if overlap_chars > 0 else end
        if start < 0:
            start = 0
    return chunks


def _encode_docs(model: SentenceTransformer, texts: List[str], batch_size: int = 64, model_name: Optional[str] = None) -> np.ndarray:
    # E5-style models prefer "passage: " prefix for documents
    docs = texts
    mname = (model_name or HE_EMBED_MODEL).lower()
    if "e5" in mname:
        docs = [f"passage: {t}" for t in texts]
    return model.encode(docs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def _load_structured_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_ocr_corpus(
    ocr_dir: str,
    embed_dir_name: str = "ocr_embeddings",
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    model_name: Optional[str] = None,
    enable_multi_vector: bool = True,
) -> Tuple[str, str]:
    """
    Build chunks and embeddings from structured OCR JSONL in `ocr_dir`.
    If `structured_documents.jsonl` is missing, create it from `results.json`.
    Returns (index.faiss path, meta.jsonl path)
    """
    jsonl_path = os.path.join(ocr_dir, "structured_documents.jsonl")
    results_json = os.path.join(ocr_dir, "results.json")
    if not os.path.exists(jsonl_path):
        if not os.path.exists(results_json):
            raise FileNotFoundError("results.json not found. Run OCR first.")
        build_ocr_documents(results_json, ocr_dir)

    documents = _load_structured_jsonl(jsonl_path)
    if not documents:
        raise RuntimeError("No documents found to index.")

    embed_dir = os.path.join(ocr_dir, embed_dir_name)
    _ensure_dir(embed_dir)

    # Build chunks list and metadata rows
    chunk_texts: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    for doc in documents:
        text = _denoise_text(doc.get("text") or "")
        if not text:
            continue
        chunks = _chunk_text(text, target_chars=chunk_chars, overlap_chars=overlap_chars)

        # Optional multi-vector fields per document: title, keywords, summary
        if enable_multi_vector:
            title_parts: List[str] = []
            if doc.get("document_type"):
                title_parts.append(str(doc.get("document_type")))
            if doc.get("issuer"):
                title_parts.append(str(doc.get("issuer")))
            if doc.get("patient_name"):
                title_parts.append(str(doc.get("patient_name")))
            if doc.get("document_date_il") or doc.get("document_date"):
                title_parts.append(str(doc.get("document_date_il") or doc.get("document_date")))
            title_parts.append(str(doc.get("category") or ""))
            title_text = _denoise_text(" | ".join([p for p in title_parts if p]))
            if title_text.strip():
                meta_rows.append({
                    "row_id": None,
                    "document_id": doc.get("document_id"),
                    "original_id": doc.get("original_id"),
                    "chron_id": doc.get("chron_id"),
                    "document_type": doc.get("document_type"),
                    "category": doc.get("category"),
                    "document_date": doc.get("document_date"),
                    "document_date_il": doc.get("document_date_il"),
                    "date_start": doc.get("date_start"),
                    "date_end": doc.get("date_end"),
                    "sort_date": doc.get("sort_date"),
                    "date_start_il": doc.get("date_start_il"),
                    "date_end_il": doc.get("date_end_il"),
                    "sort_date_il": doc.get("sort_date_il"),
                    "issuer": doc.get("issuer"),
                    "pages": doc.get("pages"),
                    "source": doc.get("source"),
                    "patient_name": doc.get("patient_name"),
                    "found_dates": doc.get("found_dates"),
                    "detected_keywords": doc.get("detected_keywords"),
                    "chunk_index": int(-1),
                    "source_field": "title",
                    "text": title_text,
                })
                chunk_texts.append(title_text)

            kw_list = [k for k in (doc.get("detected_keywords") or []) if isinstance(k, str) and k.strip()]
            kw_text = _denoise_text(", ".join(sorted(set(kw_list))))
            if kw_text.strip():
                meta_rows.append({
                    "row_id": None,
                    "document_id": doc.get("document_id"),
                    "original_id": doc.get("original_id"),
                    "chron_id": doc.get("chron_id"),
                    "document_type": doc.get("document_type"),
                    "category": doc.get("category"),
                    "document_date": doc.get("document_date"),
                    "document_date_il": doc.get("document_date_il"),
                    "date_start": doc.get("date_start"),
                    "date_end": doc.get("date_end"),
                    "sort_date": doc.get("sort_date"),
                    "date_start_il": doc.get("date_start_il"),
                    "date_end_il": doc.get("date_end_il"),
                    "sort_date_il": doc.get("sort_date_il"),
                    "issuer": doc.get("issuer"),
                    "pages": doc.get("pages"),
                    "source": doc.get("source"),
                    "patient_name": doc.get("patient_name"),
                    "found_dates": doc.get("found_dates"),
                    "detected_keywords": doc.get("detected_keywords"),
                    "chunk_index": int(-2),
                    "source_field": "keywords",
                    "text": kw_text,
                })
                chunk_texts.append(kw_text)

            if chunks:
                summary_text = (chunks[0] or "")[:500]
                if summary_text.strip():
                    meta_rows.append({
                        "row_id": None,
                        "document_id": doc.get("document_id"),
                        "original_id": doc.get("original_id"),
                        "chron_id": doc.get("chron_id"),
                        "document_type": doc.get("document_type"),
                        "category": doc.get("category"),
                        "document_date": doc.get("document_date"),
                        "document_date_il": doc.get("document_date_il"),
                        "date_start": doc.get("date_start"),
                        "date_end": doc.get("date_end"),
                        "sort_date": doc.get("sort_date"),
                        "date_start_il": doc.get("date_start_il"),
                        "date_end_il": doc.get("date_end_il"),
                        "sort_date_il": doc.get("sort_date_il"),
                        "issuer": doc.get("issuer"),
                        "pages": doc.get("pages"),
                        "source": doc.get("source"),
                        "patient_name": doc.get("patient_name"),
                        "found_dates": doc.get("found_dates"),
                        "detected_keywords": doc.get("detected_keywords"),
                        "chunk_index": int(-3),
                        "source_field": "summary",
                        "text": summary_text,
                    })
                    chunk_texts.append(summary_text)

        for idx, ch in enumerate(chunks):
            row = {
                "row_id": None,  # to be filled after indexing
                "document_id": doc.get("document_id"),
                "original_id": doc.get("original_id"),
                "chron_id": doc.get("chron_id"),
                "document_type": doc.get("document_type"),
                "category": doc.get("category"),
                "document_date": doc.get("document_date"),
                "document_date_il": doc.get("document_date_il"),
                "date_start": doc.get("date_start"),
                "date_end": doc.get("date_end"),
                "sort_date": doc.get("sort_date"),
                "date_start_il": doc.get("date_start_il"),
                "date_end_il": doc.get("date_end_il"),
                "sort_date_il": doc.get("sort_date_il"),
                "issuer": doc.get("issuer"),
                "pages": doc.get("pages"),
                "source": doc.get("source"),
                "patient_name": doc.get("patient_name"),
                "found_dates": doc.get("found_dates"),
                "detected_keywords": doc.get("detected_keywords"),
                "chunk_index": int(idx),
                "source_field": "body",
                "text": ch,
            }
            meta_rows.append(row)
            chunk_texts.append(ch)

    if not chunk_texts:
        raise RuntimeError("No chunk texts produced.")

    model = _load_model(model_name=model_name)
    embeddings = _encode_docs(model, chunk_texts, batch_size=64, model_name=model_name)
    dim = embeddings.shape[1]

    # Build FAISS index (normalized vectors; inner product == cosine)
    index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexIDMap2(index)
    ids = np.arange(len(embeddings), dtype=np.int64)
    id_index.add_with_ids(embeddings, ids)
    faiss.write_index(id_index, _index_path(embed_dir))

    with open(_meta_path(embed_dir), "w", encoding="utf-8") as f:
        for i, row in enumerate(meta_rows):
            row_out = dict(row)
            row_out["row_id"] = int(i)
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    with open(_model_info_path(embed_dir), "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name or HE_EMBED_MODEL,
            "dim": int(dim),
            "device": _device(),
            "multi_vector": bool(enable_multi_vector),
        }, f, ensure_ascii=False, indent=2)

    np.save(_vectors_path(embed_dir), embeddings)

    return _index_path(embed_dir), _meta_path(embed_dir)


def search_ocr_index(
    ocr_dir: str,
    query_text: str,
    top_k: int = 8,
    embed_dir_name: str = "ocr_embeddings",
    filters: Optional[Dict[str, Any]] = None,
    reconstruct_document_from_multivector: bool = True,
) -> List[Dict[str, Any]]:
    embed_dir = os.path.join(ocr_dir, embed_dir_name)
    index = faiss.read_index(_index_path(embed_dir))
    meta: List[Dict[str, Any]] = []
    with open(_meta_path(embed_dir), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))

    # Load the same model used for indexing if available
    model_name: Optional[str] = None
    try:
        with open(_model_info_path(embed_dir), "r", encoding="utf-8") as f:
            info = json.load(f)
            model_name = info.get("model")
    except Exception:
        model_name = None

    model = _load_model(model_name=model_name)
    q = query_text
    mname = (model_name or HE_EMBED_MODEL).lower()
    if "e5" in mname:
        # Augment query with domain synonyms to improve recall
        aug = ""
        qt = (query_text or "")
        if any(k in qt for k in ["אישור מחלה", "חופ", "תוקף", "כושר עבודה"]):
            aug = " חופש מחלה אישור מחלה תוקף עד מנוחה מעבודה"
        if any(k in qt for k in ["מצבו התפקודי", "תפקודי", "חזרה לעבודה"]):
            aug += " טווחי תנועה הגבלה כאבים NV תקינה אינו יכול כושר עבודה"
        q = f"query: {query_text}{aug}"
    # Enforce derived filters based on query intent if none provided
    ql = (query_text or "")
    if not filters:
        derived: Dict[str, Any] = {}
        if any(k in ql for k in ["אישור מחלה", "חופש", "תוקף"]):
            # Allow multiple sick-leave related document types
            derived["document_type"] = [
                "תעודה רפואית לנפגע בעבודה",
                "תעודה רפואית נוספת לנפגע בעבודה",
                "טופס סיכום בדיקה רפואית (רפואה תעסוקתית)",
            ]
        if any(k in ql for k in ["חזרה לעבודה", "תעסוק"]):
            derived["category"] = "ביקור רופא תעסוקתי"
        if derived:
            filters = derived

    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, top_k * 50)
    flat_ids = ids[0].tolist()
    flat_scores = scores[0].tolist()

    # Build quick lookup maps
    rowid_to_meta: Dict[int, Dict[str, Any]] = {}
    doc_to_body_rows: Dict[str, List[int]] = {}
    for m in meta:
        rid = int(m.get("row_id", -1))
        rowid_to_meta[rid] = m
        if (m.get("source_field") or "body") == "body":
            did = str(m.get("document_id"))
            doc_to_body_rows.setdefault(did, []).append(rid)

    # Apply simple filters on metadata if provided (supports OR lists)
    def _match(m: Dict[str, Any]) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if m.get(k) not in v:
                    return False
            else:
                if m.get(k) != v:
                    return False
        return True

    # Optionally reconstruct to a body chunk when the hit is from non-body field
    selected_by_doc: Dict[str, Dict[str, Any]] = {}
    # Preload vectors if reconstruction requested
    vectors: Optional[np.ndarray] = None
    if reconstruct_document_from_multivector:
        try:
            vectors = np.load(_vectors_path(embed_dir))
        except Exception:
            vectors = None

    for rank, (rid, sc) in enumerate(zip(flat_ids, flat_scores)):
        m = rowid_to_meta.get(int(rid))
        if not m:
            continue
        if not _match(m):
            continue
        did = str(m.get("document_id"))
        src_field = (m.get("source_field") or "body")
        out_row = m
        out_score = float(sc)
        if reconstruct_document_from_multivector and src_field != "body":
            body_rows = doc_to_body_rows.get(did) or []
            if body_rows:
                if vectors is not None and len(vectors) > max(body_rows):
                    mat = vectors[np.array(body_rows, dtype=np.int64)]  # normalized
                    sims = (mat @ q_emb[0]).astype(np.float32)
                    best_idx = int(np.argmax(sims))
                    out_row = rowid_to_meta.get(int(body_rows[best_idx])) or m
                    out_score = float(max(out_score, float(sims[best_idx])))
                else:
                    out_row = rowid_to_meta.get(int(body_rows[0])) or m
        prev = selected_by_doc.get(did)
        if (prev is None) or (out_score > prev["__score__"]):
            out_copy = dict(out_row)
            out_copy["__score__"] = out_score
            out_copy["__rank__"] = rank
            out_copy["__matched_field__"] = src_field
            selected_by_doc[did] = out_copy

    # Keyword gating and intent strictness
    kw_regex = None
    intent = None
    if any(kw in ql for kw in ["אישור מחלה", "חופש", "בתוקף"]):
        intent = "sick_leave"
        # Broaden sick-leave signal phrases
        kw_regex = re.compile(
            r"תוקף\s*עד|עד\s*תאריך|אישור\s*מחלה|אישורי\s*מחלה|ימי\s*מחלה|חופש\s*מחלה",
            re.IGNORECASE,
        )
    elif any(kw in ql for kw in ["חזרה לעבודה", "תעסוק"]):
        intent = "return_to_work"
        kw_regex = re.compile(r"אינו\s*יכול\s*לחזור|כושר\s*עבודה|חזרה\s*לעבודה", re.IGNORECASE)
    elif any(kw in ql for kw in ["מצבו התפקודי", "תפקודי"]):
        intent = "functional_status"
        kw_regex = re.compile(r"כאבים|הגבלה|טווחי\s*תנועה|אגרוף|NV\s*תקינה|פיזיוטר?פיה", re.IGNORECASE)

    def recency_key(d: Dict[str, Any]) -> str:
        return d.get("sort_date") or d.get("document_date") or "0000-00-00"

    def has_kw(d: Dict[str, Any]) -> bool:
        if not kw_regex:
            return False
        txt = d.get("text") or ""
        return bool(kw_regex.search(txt))

    # Boost per-question intents and re-rank using a compact scoring function
    def _boost_score(r: Dict[str, Any]) -> float:
        base = float(r.get("__score__", 0.0))
        txt = (r.get("text") or "")
        typ = (r.get("document_type") or "")
        cat = (r.get("category") or "")
        # Sick leave validity: certificates get a strong boost
        if intent == "sick_leave":
            if "תעודה רפואית" in typ or "אישור" in typ:
                base += 0.35
            if re.search(r"תוקף\s*עד|עד\s*תאריך", txt):
                base += 0.2
        # Return-to-work: occupational visits boosted
        if intent == "return_to_work":
            if "טופס סיכום בדיקה רפואית" in typ and ("תעסוקת" in typ or "תעסוקת" in cat):
                base += 0.35
            if re.search(r"אינו\s*יכול\s*לחזור|כושר\s*עבודה|חזרה\s*לעבודה|הגבלה\s*תעסוקתית", txt):
                base += 0.2
        # Functional status: physio summaries and function vocab boosted
        if intent == "functional_status":
            if "סיכום טיפול" in typ or "פיזיו" in typ or "פיזיו" in txt:
                base += 0.25
            if re.search(r"טווחי\s*תנועה|כאבים|NV\s*תקינה|הגבלה|אגרוף", txt):
                base += 0.15
        # Normalize to [0,1]
        return max(0.0, min(1.0, base))

    results: List[Dict[str, Any]] = []
    for r in selected_by_doc.values():
        r2 = dict(r)
        r2["__score__"] = _boost_score(r)
        results.append(r2)
    # Two-stage sort: primary score desc, then keyword presence, then recency, then original rank
    results.sort(key=lambda r: int(r.get("__rank__", 10**9)))  # stable base
    results.sort(key=lambda r: ("9999-99-99" > recency_key(r)), reverse=True)
    results.sort(key=lambda r: (has_kw(r)), reverse=True)
    results.sort(key=lambda r: float(r.get("__score__", 0.0)), reverse=True)

    # Strict intent filtering: drop results that don't match required types/phrases
    def _strict_match(d: Dict[str, Any]) -> bool:
        tt = (d.get("document_type") or "")
        cc = (d.get("category") or "")
        txt = d.get("text") or ""
        if intent == "sick_leave":
            # Accept classic certificates, occupational summaries referencing sick-leave,
            # or any document that carries explicit sick-leave date ranges in metadata.
            is_cert_type = (
                ("תעודה רפואית" in tt)
                or ("אישור" in tt)
                or ("טופס סיכום בדיקה" in tt)
                or ("תעסוקת" in tt)
            )
            has_dates_metadata = bool(d.get("date_start") or d.get("date_end"))
            has_textual_signal = bool(
                re.search(r"תוקף\s*עד|עד\s*תאריך", txt)
                or ("אישורי מחלה" in txt)
                or ("אישור מחלה" in txt)
                or ("ימי מחלה" in txt)
                or ("חופש מחלה" in txt)
            )
            return bool(is_cert_type and (has_textual_signal or has_dates_metadata))
        if intent == "return_to_work":
            if "טופס סיכום בדיקה רפואית" not in tt:
                return False
            if "תעסוקת" not in tt and "תעסוקת" not in cc:
                return False
            return bool(re.search(r"אינו\s*יכול\s*לחזור|כושר\s*עבודה|חזרה\s*לעבודה", txt))
        if intent == "functional_status":
            if not ("סיכום טיפול" in tt or "ביקור מרפאה" in tt or "פיזיו" in tt):
                # allow if strong function vocab in text
                if not re.search(r"טווחי\s*תנועה|כאבים|הגבלה|אגרוף", txt):
                    return False
            return True
        return True

    if intent:
        results = [r for r in results if _strict_match(r)]

    # Backfill with keyword-strong candidates if we still lack enough
    if kw_regex and len(results) < top_k:
        existing_doc_ids = {str(r.get("document_id")) for r in results}
        kw_candidates: List[Dict[str, Any]] = []
        for m in meta:
            if (m.get("source_field") or "body") != "body":
                continue
            if str(m.get("document_id")) in existing_doc_ids:
                continue
            if not _match(m):
                continue
            if has_kw(m):
                if not intent or _strict_match(m):
                    kw_candidates.append(m)
        kw_candidates.sort(key=lambda r: ("9999-99-99" > recency_key(r)))
        for kc in kw_candidates:
            results.append(kc)
            if len(results) >= top_k:
                break

    # Strip helper fields
    cleaned: List[Dict[str, Any]] = []
    for r in results[:top_k]:
        if "__score__" in r:
            r = {k: v for k, v in r.items() if not k.startswith("__")}
        cleaned.append(r)
    return cleaned


__all__ = [
    "index_ocr_corpus",
    "search_ocr_index",
]


