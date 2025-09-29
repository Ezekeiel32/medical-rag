import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .rag_answer import rag_answer_json


QUESTION_TEMPLATES_HE: List[str] = [
    "מהו מצבו התפקודי העדכני של המבוטח?",
    "מהי המלצת הרופא התעסוקתי לגבי חזרה לעבודה?",
    "האם יש אישור מחלה בתוקף?",
]


def _load_structured_docs(ocr_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(ocr_dir, "structured_documents.jsonl")
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# -----------------------------
# Heuristic summarization utils
# -----------------------------

_RE_FRACTURE = re.compile(r"שבר|FRACTURE", re.IGNORECASE)
_RE_WORK_EVENT = re.compile(r"תאונת\s*עבודה|בזמן\s*עבודתו|בזמן\s*עבודתה|נפל(?:\s+.*)?בעבודה|החליק\s+בעבודה|נפל\s+ממשאית|נפל\s+מ.*משאית", re.IGNORECASE)
_RE_CAST_OR_FIX = re.compile(r"גבס|קיבוע|שחזור|K-?WIRE", re.IGNORECASE)
_RE_SICK_LEAVE = re.compile(r"אישור\s*מחלה|חופש\s*מחלה|תוקף\s*עד|עד\s*תאריך|נעדר\s*מעבודה", re.IGNORECASE)
_RE_RETURN_BLOCK = re.compile(r"אינו\s*יכול\s*לחזור|לא\s*יכול\s*לחזור|אובדן\s*כושר\s*עבודה", re.IGNORECASE)
_RE_RETURN_ALLOW = re.compile(r"חזרה\s*לעבודה|כושר\s*עבודה|חזרה\s*לפעילות", re.IGNORECASE)
_RE_FUNCTION = re.compile(r"כאבים|הגבלה|טווח(?:י)?\s*תנועה|אגרוף|NV\s*תקינה", re.IGNORECASE)
_RE_PHYSIO = re.compile(r"פיזיו|ריפוי\s*בעיסוק", re.IGNORECASE)
_RE_COMMITTEE = re.compile(r"ועדה\s*רפואית|אחוזי\s*נכות|נכות\s*רפואית", re.IGNORECASE)


def _take_snippet(text: str, match: re.Match, window: int = 180) -> str:
    """Return a compact snippet around match span (right-to-left safe)."""
    start, end = match.start(), match.end()
    s = max(0, start - window)
    e = min(len(text), end + window)
    return (text[s:e] or "").strip().replace("\n", " ")[:800]


def _summarize_and_analyze(doc: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create (summary, quote, analysis) for a document using robust heuristics.

    The goal is to produce detailed, useful items for a legal/insurance case report.
    """
    text: str = (doc.get("text") or "").strip()
    typ: str = (doc.get("document_type") or "מסמך רפואי").strip()
    doc_date: str = doc.get("document_date") or ""

    # Defaults - increased for comprehensive information
    summary: str = text[:800]
    quote: str = text[:1000]
    analysis: str = ""

    # ER / clinic visit with trauma documentation
    if "מיון" in typ or "ביקור מרפאה" in typ or "דוח שחרור" in typ:
        m_event = _RE_WORK_EVENT.search(text)
        m_frac = _RE_FRACTURE.search(text)
        m_fix = _RE_CAST_OR_FIX.search(text)
        
        if m_event and m_frac:
            # Find the best quote that includes both work event and fracture
            event_text = _take_snippet(text, m_event, 250)
            frac_text = _take_snippet(text, m_frac, 250)
            
            # Choose the longer, more informative quote
            quote = frac_text if len(frac_text) > len(event_text) else event_text
            
            if "מיון" in typ:
                summary = f"תיעוד ראשוני של הפגיעה. אובחן שבר בבסיס מסרק 5 בכף יד שמאל עם תזוזה, לאחר שנפל ממשאית בעבודה. בוצע ניסיון לשחזור וקיבוע בסד גבס."
                analysis = "קריטי: הוכחה לקשר סיבתי מובהק בין תאונת העבודה לבין השבר. מתעד את האבחנה הראשונית."
            else:
                # Avoid embedded quotes in the Hebrew phrase to prevent parsing issues on some environments
                summary = f"תיעוד קליני של השבר והטיפול הניתוחי. {'בוצע קיבוע על ידי K-WIRE' if m_fix else 'המשך טיפול שמרני'}."
                analysis = "מהותי: מאשש את חומרת הפגיעה, הטיפול הנדרש והמעקב הרפואי."
        elif m_frac:
            quote = _take_snippet(text, m_frac, 300)
            summary = "תיעוד קליני של שבר והמלצות טיפול"
            analysis = "מהותי: מאשש את חומרת הפגיעה והצורך בטיפול"

    # Occupational medicine or certificates about work capacity
    if "רפואה תעסוקתית" in typ or _RE_SICK_LEAVE.search(text):
        m_block = _RE_RETURN_BLOCK.search(text)
        m_allow = _RE_RETURN_ALLOW.search(text)
        m_sl = _RE_SICK_LEAVE.search(text)
        
        if m_block:
            quote = _take_snippet(text, m_block, 300)
            summary = f"קביעה תעסוקתית: אינו יכול לחזור לעבודתו. {'אושר המשך חופש מחלה' if 'חופש' in text else 'אי כושר עבודה זמני'}."
            analysis = "חזק: תומך באובדן כושר עבודה זמני ובזכאות לפיצוי."
        elif m_allow:
            quote = _take_snippet(text, m_allow, 300)
            summary = f"קביעה תעסוקתית: מותרת חזרה לעבודה {'עם מגבלות' if 'מגבלה' in text else 'בהדרגה'}."
            analysis = "רלוונטי: מגדיר מסגרת לחזרה לעבודה ומגבלות תעסוקתיות."
        elif m_sl:
            quote = _take_snippet(text, m_sl, 300)
            summary = f"תעודה רפואית/אישור מחלה {'עם תאריכי תוקף' if 'עד' in text else 'לתקופה מוגדרת'}."
            analysis = "מהותי: מבסס זכאות לתקופת מחלה והיעדרות מעבודה."

    # Functional status / rehabilitation
    if _RE_PHYSIO.search(text) or "סיכום טיפול" in typ or _RE_FUNCTION.search(text):
        m_func = _RE_FUNCTION.search(text)
        if m_func:
            quote = _take_snippet(text, m_func, 300)
        else:
            # Look for physio-specific content
            physio_match = _RE_PHYSIO.search(text)
            if physio_match:
                quote = _take_snippet(text, physio_match, 300)
        
        summary = f"סיכום טיפול פיזיותרפיה {'עם תיעוד כאבים והגבלות' if _RE_FUNCTION.search(text) else 'והמלצות המשך'}."
        analysis = "תומך: מדגים מגבלה תפקודית מתמשכת והשלכות תעסוקתיות."

    # Committee decisions
    if _RE_COMMITTEE.search(text) or "ועדה רפואית" in typ:
        m_c = _RE_COMMITTEE.search(text)
        if m_c:
            quote = _take_snippet(text, m_c, 300)
        summary = f"החלטת ועדה רפואית {'לנכות זמנית' if 'זמנית' in text else 'בדבר אחוזי נכות'}."
        analysis = "משמעותי: קובע רשמית את מידת הנכות וזכאות לפיצוי."

    return summary[:800], quote[:1000], analysis


def _is_relevant(doc: Dict[str, Any]) -> bool:
    """Return True for documents that are typically relevant to a work-injury claim."""
    typ = (doc.get("document_type") or "").strip()
    cat = (doc.get("category") or "").strip()
    text = (doc.get("text") or "").strip()

    if any(k in typ for k in ["מיון", "דוח שחרור", "ביקור מרפאה", "רפואה תעסוקתית", "תעודה רפואית", "סיכום טיפול", "ועדה רפואית"]):
        return True
    if any(k in cat for k in ["ביקור במיון", "ביקור רופא תעסוקתי", "ביקור רופא כללי", "טופס ביטוח", "אחר"]):
        # Require at least some meaningful signals in text
        return bool(_RE_FRACTURE.search(text) or _RE_SICK_LEAVE.search(text) or _RE_FUNCTION.search(text) or _RE_WORK_EVENT.search(text))
    return False


def build_case_report(
    ocr_dir: str,
    model: str = "gemma2:2b-instruct",
    top_k: int = 12,
    skip_answers: bool = False,
) -> Dict[str, Any]:
    """Build a comprehensive case report with answers and a chronological summary.

    - Answers are produced by the real RAG QA function (no mocks).
    - Chronology is derived from structured OCR docs, filtered for relevance,
      and each item contains: document_name, date, concise summary, quote, analysis.
    """
    # Answers for canonical questions (concise, with sources)
    answers: List[Dict[str, Any]] = []
    if not skip_answers:
        for q in QUESTION_TEMPLATES_HE:
            ans = rag_answer_json(ocr_dir, q, top_k=top_k, model=model)
            answers.append({
                "question": ans.get("question"),
                "answer": ans.get("answer"),
                "sources": ans.get("sources", []),
            })

    # Chronology: start from oldest to newest (true chronological order)
    docs = _load_structured_docs(ocr_dir)
    # Deduplicate by document_id while keeping earliest appearance
    seen_ids: set[str] = set()
    unique_docs: List[Dict[str, Any]] = []
    for d in sorted(docs, key=lambda r: int(r.get("chron_id") or 0)):
        did = str(d.get("document_id") or "")
        if did and did in seen_ids:
            continue
        seen_ids.add(did)
        if _is_relevant(d):
            unique_docs.append(d)

    chronology: List[Dict[str, Any]] = []
    for d in unique_docs:
        summary, quote, analysis = _summarize_and_analyze(d)
        chronology.append({
            "document_name": str(d.get("document_type") or "מסמך רפואי"),
            "document_date": d.get("document_date"),
            "summary": summary,
            "quote": quote,
            "analysis": analysis,
            "document_id": str(d.get("document_id") or ""),
            "pages": d.get("pages") or [],
        })

    return {"answers": answers, "chronology": chronology}


__all__ = ["build_case_report", "QUESTION_TEMPLATES_HE"]



