from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
from datetime import datetime, date, timedelta

from .rag_ocr import search_ocr_index
from .query_intelligence import create_intelligent_rag_response
from .llm_ollama import ollama_generate
from .llm_gemini import gemini_generate
from .timezone_utils import (
    get_jerusalem_now, get_jerusalem_today, get_jerusalem_date_string,
    parse_date_with_jerusalem_tz, is_sick_leave_valid,
    get_medical_context_relevancy, format_date_for_display
)


ANSWER_SCHEMA_EXAMPLE = {
    "question": "מהו מצבו התפקודי העדכני של המבוטח?",
    "answer": "טקסט קצר וקולע בעברית.",
    "sources": [
        {
            "document_type": "דוח ביקור מרפאה",
            "document_date": "2025-04-11",
            "pages": [6],
            "quote": "ציטוט מדויק",
            "document_id": "45119507",
            "chron_id": 34,
            "original_id": "45119507",
        }
    ]
}
def _clean_context_text(text: str) -> str:
    """Aggressive de-noising for LLM context: trim admin headers and numbers-only lines."""
    if not text:
        return ""
    # Remove very long sequences of digits/punctuation that are typical in headers
    lines = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t:
            continue
        if _RE_NOISE.search(t):
            continue
        # Drop pure numbers or phone/ID-like rows
        if not _RE_HE.search(t):
            digits = re.sub(r"[^0-9]", "", t)
            if len(digits) >= max(6, int(0.7 * len(t))):
                continue
        lines.append(t)
    out = " ".join(lines)
    out = re.sub(r"\s{2,}", " ", out)
    return out



def _needs_refine(answer: str, rows: List[Dict[str, Any]]) -> bool:
    """Heuristics to detect parroting or low-quality draft answers.

    - Very long or multi-paragraph
    - Contains admin noise like phone/ID patterns
    - Appears to copy a long prefix from the top of a source text
    """
    if not isinstance(answer, str):
        return True
    text = answer.strip()
    if not text:
        return True
    if len(text) > 700:
        return True
    if text.count("\n") >= 3:
        return True
    if re.search(r"\b\d{7,}\b|\d{2,3}-\d{6,7}|\d\s\d\s\d", text):
        return True
    if _RE_NOISE.search(text):
        return True
    try:
        first = (rows[0].get("text") or "").strip()
    except Exception:
        first = ""
    if first:
        prefix = re.sub(r"\s+", " ", first)[:100]
        if prefix and prefix in re.sub(r"\s+", " ", text):
            return True
    return False


def build_context_rows(rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for r in rows:
        raw_text = (r.get("text") or "")
        cleaned_text = _clean_context_text(raw_text)
        parts.append(json.dumps({
            "document_id": r.get("document_id"),
            "original_id": r.get("original_id"),
            "chron_id": r.get("chron_id"),
            "document_type": r.get("document_type"),
            "category": r.get("category"),
            "document_date": r.get("document_date"),
            "pages": r.get("pages"),
            "text": cleaned_text[:900],
        }, ensure_ascii=False))
    return "\n".join(parts)


SYSTEM_PROMPT_HE = (
    "אתה מומחה רפואי מנוסה המתמחה במתן תשובות מפורטות וברורות על שאלות רפואיות בעברית. "
    "המשימה שלך היא ליצור תשובה מקיפה, קוהרנטי ומובנת המבוססת על המקורות הרפואיים שסופקו. "
    "\n\nכללי יסוד:\n"
    "• ענה אך ורק על סמך המקורות שסופקו - אין להמציא או להוסיף מידע חיצוני\n"
    "• צור תשובה מלאה וקוהרנטית הכוללת את כל המידע הרלוונטי מהמקורות\n"
    "• כתוב במשפטים מלאים וברורים, ללא קטעי משפטים או רשימות לא מובנות\n"
    "• התשובה צריכה להיות מובנת וכוללת, לא מקוטעת או חסרה\n"
    "• אם אין מידע מספיק במקורות, כתוב: 'אין מידע מספיק במקורות הזמינים'\n"
    "\nדרישות עיצוב התשובה:\n"
    "• החזר JSON תקני יחיד: {\"question\",\"answer\",\"sources\"}\n"
    "• ה-answer צריך להיות סיכום מלא ומפורט (3-8 משפטים) בעברית\n"
    "• כל משפט צריך להיות שלם ומובן בפני עצמו\n"
    "• אסור להעתיק שורות שלמות מהמסמך או לכלול פרטים מנהליים\n"
    "• sources: עד 3 מקורות עם document_id, document_type, document_date, pages, quote"
)


def _today_il() -> str:
    return datetime.now().strftime("%d/%m/%Y")


def prompt_for_question(question_he: str, context_rows: str) -> str:
    # Add specific guidance based on question type
    specific_guidance = ""
    if "מצבו התפקודי" in question_he or "מצב תפקודי" in question_he:
        specific_guidance = (
            "- התמקד במצב התפקודי הנוכחי: כאבים, מגבלות תנועה, יכולת לבצע פעולות.\n"
            "- חפש מידע מהמסמכים העדכניים ביותר על מצב המטופל.\n"
            "- כלול פרטים קליניים ספציפיים כמו טווחי תנועה, כאבים, הגבלות.\n"
            "- צטט ממצאים קליניים מדויקים מהמסמכים.\n"
        )
    elif "רופא התעסוקתי" in question_he or "חזרה לעבודה" in question_he:
        specific_guidance = (
            "- חפש המלצות ספציפיות של רופא תעסוקתי על חזרה לעבודה.\n"
            "- התמקד בהמלצות לגבי כושר עבודה ומגבלות.\n"
            "- חפש תאריכי המלצה וחזרה לעבודה.\n"
            "- צטט בדיוק את המלצות הרופא התעסוקתי.\n"
        )
    elif "אישור מחלה" in question_he or "תוקף" in question_he:
        specific_guidance = (
            "- חפש תעודות מחלה עם תאריכי תוקף.\n"
            "- השווה תאריכי תוקף לתאריך הנוכחי.\n"
            "- צטט בדיוק את תאריכי התוקף מהמסמכים.\n"
        )
    
    return (
        "השאלה:\n" + question_he + "\n\n" +
        "תאריך נוכחי: " + _today_il() + "\n\n" +
        "מקורות (JSON per line):\n" + context_rows + "\n\n" +
        "הוראות יצירת תשובה מקיפה:\n"
        "- החזר אך ורק JSON אחד במבנה: {\\\"question\\\",\\\"answer\\\",\\\"sources\\\"}.\n"
        "- answer: צור תשובה מלאה, קוהרנטית ומפורטת (3-8 משפטים) בעברית טהורה.\n"
        "- כל משפט חייב להיות שלם ומובן. אין משפטים חתוכים או לא גמורים.\n"
        "- התשובה תכלול את כל המידע הרלוונטי מהמקורות בצורה מאורגנת וברורה.\n"
        "- sources: כלול ציטוטים מדויקים וקצרים עם מספרי עמודים.\n"
        "- בחר את המקורות הרלוונטיים והעדכניים ביותר לשאלה.\n"
        "- אסור לכלול טקסט מחוץ ל-JSON.\n" +
        specific_guidance
    )


_DATE_RE = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")


def _parse_he_date(text: str) -> Optional[date]:
    m = _DATE_RE.search(text)
    if not m:
        return None
    d, mth, y = re.split(r"[./-]", m.group(1))
    try:
        yi = int(y)
        if yi < 100:
            yi = 2000 + yi if yi <= 79 else 1900 + yi
        return date(int(yi), int(mth), int(d))
    except Exception:
        return None


def _extract_sick_leave_ranges(txt: str) -> List[Tuple[Optional[date], Optional[date]]]:
    ranges: List[Tuple[Optional[date], Optional[date]]] = []
    # patterns: "מיום X עד יום Y" or "מיום X עד Y"
    for m in re.finditer(r"מ(?:יום)?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}).{0,40}?עד\s*(?:יום\s*)?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", txt, flags=re.DOTALL):
        s = _parse_he_date(m.group(1))
        e = _parse_he_date(m.group(2))
        ranges.append((s, e))
    # pattern: "תוקף עד Y"
    for m in re.finditer(r"תוקף\s*עד\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", txt):
        e = _parse_he_date(m.group(1))
        ranges.append((None, e))
    # pattern: "עד תאריך Y"
    for m in re.finditer(r"עד\s*תאריך\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", txt):
        e = _parse_he_date(m.group(1))
        ranges.append((None, e))
    return ranges


def _determine_sick_leave_valid(rows: List[Dict[str, Any]]) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    # Use Jerusalem timezone for current date
    today = get_jerusalem_today()
    sources: List[Dict[str, Any]] = []
    verdict: Optional[bool] = None

    # Prefer recent documents
    def _docdate(r: Dict[str, Any]) -> date:
        ds = r.get("document_date") or r.get("sort_date") or "0000-00-00"
        try:
            return datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            return date(1, 1, 1)

    rows_sorted = sorted(rows, key=_docdate, reverse=True)

    # Explicit ranges
    for r in rows_sorted:
        txt = (r.get("text") or "")
        rngs = _extract_sick_leave_ranges(txt)
        for s, e in rngs:
            if e and today <= e and (s is None or today >= s):
                if not sources:
                    sources.append({
                        "document_id": str(r.get("document_id") or ""),
                        "document_type": r.get("document_type"),
                        "document_date": r.get("document_date"),
                        "pages": r.get("pages"),
                        "quote": txt.strip()[:280],
                    })
                verdict = True
                break
        if verdict:
            break

    # Phrasal window: "מנוחה מעבודה ... עוד חודש" -> treat as 30 days from document_date
    if verdict is None:
        for r in rows_sorted:
            txt = (r.get("text") or "")
            if re.search(r"מנוחה\s*מעבודה.*עוד\s*חודש", txt):
                ds = r.get("document_date")
                try:
                    start = datetime.strptime(ds, "%Y-%m-%d").date() if ds else None
                except Exception:
                    start = None
                if start:
                    end = start + timedelta(days=30)
                    if today <= end:
                        sources.append({
                            "document_id": str(r.get("document_id") or ""),
                            "document_type": r.get("document_type"),
                            "document_date": r.get("document_date"),
                            "pages": r.get("pages"),
                            "quote": txt.strip()[:280],
                        })
                        return True, sources[:2]

    if verdict is None:
        # If we saw any ranges but not valid today, set False and cite the most recent end date
        latest: Optional[Tuple[date, Dict[str, Any]]] = None
        for r in rows_sorted:
            txt = (r.get("text") or "")
            rngs = _extract_sick_leave_ranges(txt)
            for s, e in rngs:
                if e:
                    if latest is None or e > latest[0]:
                        latest = (e, {
                            "document_id": str(r.get("document_id") or ""),
                            "document_type": r.get("document_type"),
                            "document_date": r.get("document_date"),
                            "pages": r.get("pages"),
                            "quote": txt.strip()[:280],
                        })
        if latest:
            verdict = False
            if not sources:
                sources.append(latest[1])
    return verdict, sources


def _row_docdate(r: Dict[str, Any]) -> date:
    ds = r.get("document_date") or r.get("sort_date") or "0000-00-00"
    try:
        return datetime.strptime(ds, "%Y-%m-%d").date()
    except Exception:
        return date(1, 1, 1)


def _take_snippet(text: str, start: int, end: int, window: int = 160) -> str:
    if start < 0 or end < 0:
        base = (text or "").strip()[:280]
        cleaned = _clean_snippet_text(base)
        if len(cleaned) < 30:
            alt = _best_clinical_snippet(text)
            if alt:
                return alt
        return cleaned or base
    s = max(0, start - window)
    e = min(len(text), end + window)
    base = text[s:e]
    # Focus on clinical part if present within the window
    m = _RE_CLINICAL.search(base)
    if m and m.start() > 0:
        base = base[max(0, m.start() - 8):]
    base = base.strip()[:360]
    cleaned = _clean_snippet_text(base)
    if len(cleaned) < 30:
        alt = _best_clinical_snippet(text)
        if alt:
            return alt
    return cleaned or base


_RE_HE = re.compile(r"[א-ת]")
_RE_NOISE = re.compile(
    r"שם\s*משפחה|שם\s*פרטי|טלפון|מס(?:פר)?\s*זהות|ת\.?ז\.?|מס'|כתובת|קופה|מרפאה|הודפס|סניף|מס' לקוח",
    re.IGNORECASE,
)


def _clean_snippet_text(snippet: str) -> str:
    """Remove administrative headers/noise so we don't parrot the top of the PDF.

    Keeps lines that contain Hebrew letters or clinical keywords, drops lines that are
    numbers-only or typical admin labels (phone, ID, names). Returns a compact string.
    """
    if not snippet:
        return ""
    lines = [ln.strip() for ln in snippet.splitlines()]
    kept: list[str] = []
    for ln in lines:
        if not ln or len(ln) <= 2:
            continue
        if _RE_NOISE.search(ln):
            continue
        # Drop lines that are almost only digits/punctuation and have no Hebrew
        if not _RE_HE.search(ln):
            digits = re.sub(r"[^0-9]", "", ln)
            if len(digits) >= max(6, int(0.7 * len(ln))):
                continue
        kept.append(ln)
    out = " ".join(kept)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:360]


_RE_CLINICAL = re.compile(
    r"הערכת\s*מטפל|כאבים|הגבלה|טווח(?:י)?\s*תנועה|אגרוף|סגירת|כיפוף|ממצאים|בדיקה|התרשמות|סיכום\s*טיפול",
    re.IGNORECASE,
)


def _best_clinical_snippet(text: str, window: int = 160) -> str:
    if not text:
        return ""
    m = _RE_CLINICAL.search(text)
    if not m:
        return ""
    s = max(0, m.start() - window)
    e = min(len(text), m.end() + window)
    base = text[s:e].strip()[:360]
    cleaned = _clean_snippet_text(base)
    return cleaned or base[:280]


def _find_initial_injury_source(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find earliest ER/clinic document that links work accident with fracture.
    Returns a minimal source dict or None.
    """
    try:
        rows_oldest_first = sorted(rows, key=_row_docdate)
    except Exception:
        rows_oldest_first = rows
    for r in rows_oldest_first:
        txt = (r.get("text") or "")
        doc_type = r.get("document_type") or ""
        if any(k in doc_type for k in ["מיון", "דוח שחרור", "ביקור מרפאה"]) or ("EM" in (r.get("document_id") or "")):
            if re.search(r"נפל|החליק|בזמן\s*עבוד", txt) and re.search(r"שבר|FRACTURE", txt):
                m_event = re.search(r"(?:בזמן\s*עבודתו?|נפל|החליק)[^\n\r]{0,200}", txt)
                m_frac = re.search(r"שבר[^\n\r]{0,160}", txt)
                joined = " ".join(part.group(0) for part in [m_event, m_frac] if part)
                snippet = joined.strip()[:360] if joined else (txt.strip()[:360])
                return {
                    "document_id": str(r.get("document_id") or ""),
                    "document_type": doc_type,
                    "document_date": r.get("document_date"),
                    "pages": r.get("pages"),
                    "quote": snippet,
                }
    return None


def _determine_functional_status(rows: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract and summarize the current functional status from medical documents.
    Returns a comprehensive summary of the patient's current condition and limitations.
    """
    # Clinical markers for functional assessment
    core = r"כאבים|הגבלה|טווח(?:י)?\s*תנועה|אגרוף|כיפוף|סגירת|רגישות|נפיחות|דיפורמצ|הרמת\s*חפצים|שיפור|הטבה|ללא\s*הטבה"
    keys = re.compile(core, re.IGNORECASE)
    rows_sorted = sorted(rows, key=_row_docdate, reverse=True)

    # Find the most recent comprehensive assessment
    for r in rows_sorted:
        txt = (r.get("text") or "")
        doc_type = r.get("document_type") or ""
        
        # Look for comprehensive functional assessments
        if any(term in txt for term in ["הערכת מטפל", "ממצאים", "סיכום טיפול", "בדיקה"]):
            m = keys.search(txt)
            if m:
                # Extract a comprehensive context around clinical findings - increased for full information
                snippet = _take_snippet(txt, max(0, m.start() - 100), m.end() + 300)
                if len(snippet) < 30:
                    continue
                    
                d_il = r.get("document_date")
                try:
                    d_il = format_date_for_display(datetime.strptime(d_il, "%Y-%m-%d").date()) if d_il else None
                except Exception:
                    d_il = None

                # Build comprehensive status summary based on findings
                status_parts = []
                
                # Pain assessment
                if re.search(r"כאבים", txt):
                    if re.search(r"דרוג.*[5-9]|כאבים.*חזקים|כאבים.*קשים", txt):
                        status_parts.append("כאבים משמעותיים")
                    elif re.search(r"כאבים.*קלים|דרוג.*[1-3]", txt):
                        status_parts.append("כאבים קלים")
                    else:
                        status_parts.append("כאבים")
                
                # Functional limitations
                limitations = []
                if re.search(r"הגבלה.*אגרוף|חוסר.*סגירת|הגבלה.*כיפוף", txt):
                    limitations.append("הגבלה בתנועת האצבעות")
                if re.search(r"הרמת\s*חפצים|הרמה.*מוגבלת", txt):
                    limitations.append("מגבלות בהרמת חפצים")
                if re.search(r"טווח.*תנועה.*מוגבל|טווחי.*תנועה.*הגבלה", txt):
                    limitations.append("הגבלה בטווח התנועה")
                
                if limitations:
                    status_parts.append(f"מגבלות תפקודיות: {', '.join(limitations)}")
                
                # Improvement/progress assessment
                if re.search(r"שיפור.*חלקי", txt):
                    status_parts.append("שיפור חלקי בטיפול")
                elif re.search(r"ללא.*הטבה|ללא.*שיפור", txt):
                    status_parts.append("ללא שיפור משמעותי")
                
                # Construct comprehensive answer with complete information
                # Sanitize stray non-lexical tokens like standalone single letters (e.g., "אה")
                def _sanitize(text: str) -> str:
                    # Remove isolated 1-2 char Hebrew tokens (like "אה")
                    text = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", text)
                    # Remove common OCR artifacts
                    text = re.sub(r"\s+[,\.]+", ".", text)
                    # Collapse multiple spaces and clean punctuation
                    text = re.sub(r"\s{2,}", " ", text).strip()
                    # Remove standalone punctuation
                    text = re.sub(r"\s+[,\.]\s*$", ".", text)
                    return text
                if status_parts:
                    main_status = _sanitize(". ".join(status_parts))
                    
                    # Add more clinical detail extraction
                    additional_details = []
                    if re.search(r"NV.*תקינה|NV\s*תקינה", txt):
                        additional_details.append("בדיקה נוירווסקולרית תקינה")
                    if re.search(r"נפחות|דיפורמציה", txt):
                        additional_details.append("ללא נפיחות או דיפורמציה")
                    if re.search(r"רגישות", txt):
                        additional_details.append("רגישות קלה")
                    if re.search(r"מנוחה.*עבודה|חופש.*מחלה", txt):
                        work_rest_match = re.search(r"מנוחה.*עבודה.*(?:עוד|עד).*?(חודש|שבועות?|ימים?)", txt)
                        if work_rest_match:
                            additional_details.append(f"מנוחה מעבודה {work_rest_match.group()}")
                    
                    detail_text = _sanitize(". ".join(additional_details)) if additional_details else ""
                    answer = _sanitize(f"נכון ל-{d_il}: {main_status}. על פי {doc_type}, המטופל מציג {snippet.strip()}. {detail_text}").strip()
                else:
                    answer = f"נכון ל-{d_il}: המבוטח עדיין מציג מגבלות תפקודיות. על פי {doc_type}: \"{snippet}\""
                
                src = [{
                    "document_id": str(r.get("document_id") or ""),
                    "document_type": doc_type,
                    "document_date": r.get("document_date"),
                    "pages": r.get("pages"),
                    "quote": snippet[:1000],  # Increased quote length for comprehensive information
                }]
                return answer, src
    
    # Fallback: basic functional status if comprehensive assessment not found
    for r in rows_sorted:
        txt = (r.get("text") or "")
        if any(term in txt for term in ["כאבים", "הגבלה", "מגבלה"]):
            m = keys.search(txt)
            if m:
                snippet = _take_snippet(txt, max(0, m.start() - 50), m.end() + 200)  # Increased snippet size
                if len(snippet) < 20:
                    continue
                    
                d_il = r.get("document_date")
                try:
                    d_il = format_date_for_display(datetime.strptime(d_il, "%Y-%m-%d").date()) if d_il else None
                except Exception:
                    d_il = None
                    
                doc_type = r.get("document_type") or ""
                # Sanitize snippet to remove stray non-lexical tokens
                snippet_clean = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", snippet)
                snippet_clean = re.sub(r"\s{2,}", " ", snippet_clean).strip()
                answer = f"נכון ל-{d_il}: המבוטח מציג תסמינים תפקודיים. על פי {doc_type}: \"{snippet_clean}\""
                
                src = [{
                    "document_id": str(r.get("document_id") or ""),
                    "document_type": doc_type,
                    "document_date": r.get("document_date"),
                    "pages": r.get("pages"),
                    "quote": snippet[:1000],  # Increased quote length
                }]
                return answer, src
    
    return None, []


def _determine_return_to_work(rows: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract explicit occupational recommendation about return-to-work.

    Detects phrases like 'אינו יכול לחזור לעבודתו', 'כושר עבודה', 'מנוחה מעבודה', 'חזרה לפעילות מלאה'.
    """
    patt_block = re.compile(r"אינו\s*יכול\s*לחזור|לא\s*יכול\s*לחזור|אינו\s*כשיר|אובדן\s*כושר\s*עבודה|חופש\s*מחלה|יש\s*לאשר\s*המשך\s*חופש\s*מחלה", re.IGNORECASE)
    patt_allow = re.compile(r"חזרה\s*לעבודה|חזרה\s*לפעילות\s*מלאה|כושר\s*עבודה", re.IGNORECASE)
    patt_until = re.compile(r"(?:עד\s*תאריך|תוקף\s*עד)\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")
    rows_sorted = sorted(rows, key=_row_docdate, reverse=True)
    initial_src: Optional[Dict[str, Any]] = _find_initial_injury_source(rows)

    # First pass: look for occupational doctor documents (highest priority)
    for r in rows_sorted:
        txt = (r.get("text") or "")
        doc_type = r.get("document_type") or ""
        if "טופס סיכום בדיקה רפואית" in doc_type and ("תעסוקתי" in txt or "תעסוקת" in doc_type):
            m_block = patt_block.search(txt)
            m_allow = patt_allow.search(txt)
            if m_block or m_allow:
                until = None
                mu = patt_until.search(txt)
                if mu:
                    try:
                        d = _parse_he_date(mu.group(1))
                        if d:
                            until = d.strftime("%d/%m/%Y")
                    except Exception:
                        until = None
                snippet = _take_snippet(txt, (m_block or m_allow).start(), (m_block or m_allow).end())
                d_il = r.get("document_date")
                try:
                    d_il = format_date_for_display(datetime.strptime(d_il, "%Y-%m-%d").date()) if d_il else None
                except Exception:
                    d_il = None
                # Create richer, more detailed answer with explicit status, date, doc type, and context
                base = "אינו יכול לחזור לעבודה" if m_block else "מותרת חזרה לעבודה (בהתאם למגבלות)"
                if m_block and until:
                    base = f"אינו יכול לחזור לעבודה עד {until}"
                # Compose narrative with initial injury if present
                if initial_src and initial_src.get("document_date"):
                    try:
                        d0 = format_date_for_display(datetime.strptime(initial_src["document_date"], "%Y-%m-%d").date())
                    except Exception:
                        d0 = initial_src.get("document_date")
                    answer = f"מאירוע הפגיעה ({d0}) ועדכני ל-{d_il}: {base}. {doc_type}: \"{snippet}\""
                else:
                    answer = f"{base} ({d_il}). {doc_type}: \"{snippet}\""
                src = []
                if initial_src:
                    src.append(initial_src)
                src.append({
                    "document_id": str(r.get("document_id") or ""),
                    "document_type": r.get("document_type"),
                    "document_date": r.get("document_date"),
                    "pages": r.get("pages"),
                    "quote": snippet,
                })
                return answer, src[:2]

    # Second pass: any document with return-to-work content
    for r in rows_sorted:
        txt = (r.get("text") or "")
        m_block = patt_block.search(txt)
        m_allow = patt_allow.search(txt)
        if not (m_block or m_allow):
            continue
        until = None
        mu = patt_until.search(txt)
        if mu:
            try:
                d = _parse_he_date(mu.group(1))
                if d:
                    until = d.strftime("%d/%m/%Y")
            except Exception:
                until = None
        snippet = _take_snippet(txt, (m_block or m_allow).start(), (m_block or m_allow).end())
        d_il = r.get("document_date")
        try:
            d_il = format_date_for_display(datetime.strptime(d_il, "%Y-%m-%d").date()) if d_il else None
        except Exception:
            d_il = None
        doc_type = r.get("document_type") or ""
        # Create richer, more detailed answer with explicit status, date, doc type, and context
        if m_block:
            base = "אינו יכול לחזור לעבודה"
            if until:
                base = f"אינו יכול לחזור לעבודה עד {until}"
        else:
            base = "מותרת חזרה לעבודה (בהתאם למגבלות)"

        # Add document context and date
        # Sanitize snippet to ensure real words only (remove stray 1-2 letter tokens)
        snippet_clean = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", snippet)
        snippet_clean = re.sub(r"\s{2,}", " ", snippet_clean).strip()
        answer = f"{base} ({d_il}). {doc_type}: \"{snippet_clean}\""
        src = [{
            "document_id": str(r.get("document_id") or ""),
            "document_type": r.get("document_type"),
            "document_date": r.get("document_date"),
            "pages": r.get("pages"),
            "quote": snippet,
        }]
        return answer, src
    return None, []


def rag_answer_json(
    ocr_dir: str,
    question_he: str,
    top_k: int = 6,
    model: str = "gemma2:2b-instruct",
    category: Optional[str] = None,
    doc_type: Optional[str] = None,
    use_intelligent_system: bool = True,
) -> Dict[str, Any]:
    # Use intelligent system for enhanced responses
    if use_intelligent_system:
        # Check if this is a complex query that benefits from intelligence system
        if any(keyword in question_he for keyword in [
            'סכם', 'סיכום', 'תסכם', 'מה קרה', 'חודש', 'נולד', 'גיל', 
            'בחודש', 'התפתחות', 'התקדמות', 'לאחרונה', 'בתקופה'
        ]):
            try:
                return create_intelligent_rag_response(
                    question_he,
                    search_ocr_index,
                    ocr_dir,
                    model,
                    top_k
                )
            except Exception as e:
                # Fall back to standard system if intelligence system fails
                print(f"Intelligence system error: {e}, falling back to standard system")
                pass
    
    filters: Dict[str, Any] = {}
    if category:
        filters["category"] = category
    if doc_type:
        filters["document_type"] = doc_type
    if not filters:
        filters = None

    rows = search_ocr_index(ocr_dir, question_he, top_k=top_k, filters=filters)
    if not rows:
        return {
            "question": question_he,
            "answer": "לא נמצאו מקורות מתאימים לענות על השאלה.",
            "sources": []
        }

    # Special deterministic handlers
    def _load_page_text_map(ocr_dir_local: str) -> dict[int, str]:
        import os, json
        page_map: dict[int, str] = {}
        try:
            # Prefer cleaned then aligned then base
            for fname in ("results_aligned_cleaned.json", "results_aligned.json", "results.json"):
                path = os.path.join(ocr_dir_local, fname)
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    pages = data.get("pages") or []
                    for p in pages:
                        try:
                            page_map[int(p.get("page") or 0)] = str(p.get("text") or "")
                        except Exception:
                            continue
                    break
        except Exception:
            return {}
        return page_map

    def _enrich_sources_with_page_text(srcs: list[dict], page_map: dict[int, str]) -> list[dict]:
        if not srcs:
            return srcs
        out: list[dict] = []
        for s in srcs:
            q = s.get("quote") or ""
            pages = s.get("pages") or []
            if isinstance(pages, list) and pages:
                first_page = pages[0]
                if isinstance(first_page, int) and first_page in page_map:
                    q = page_map[first_page] or q
            s2 = dict(s)
            s2["quote"] = (q or "")[:800]
            out.append(s2)
        return out

    # Additional deterministic extractors: ID (ת"ז), birth date, claim/policy numbers, medications
    def _load_structured_docs_map(dir_path: str) -> Dict[int, Dict[str, Any]]:
        page_to_doc: Dict[int, Dict[str, Any]] = {}
        try:
            path = os.path.join(dir_path, "structured_documents.jsonl")
            if not os.path.exists(path):
                return {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    for p in (d.get("pages") or []):
                        try:
                            page_to_doc[int(p)] = d
                        except Exception:
                            pass
        except Exception:
            return {}
        return page_to_doc

    def _normalize_il_id(raw_id: str) -> Optional[str]:
        digits = "".join(ch for ch in raw_id if ch.isdigit())
        if not digits:
            return None
        digits = digits.zfill(9)
        if len(digits) != 9:
            return None
        total = 0
        for i, ch in enumerate(digits):
            n = int(ch) * (1 if i % 2 == 0 else 2)
            if n > 9:
                n -= 9
            total += n
        if total % 10 != 0:
            return None
        return digits

    def _extract_il_id_from_text(text: str) -> List[str]:
        cands: List[str] = []
        patterns = [
            r"תעודת\s*זהות\s*[:\-]?\s*([0-9\-]{5,12})",
            r"ת\.?\s*ז\.?\s*[:\-]?\s*([0-9\-]{5,12})",
            r"מס'?\s*זהות\s*[:\-]?\s*([0-9\-]{5,12})",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                norm = _normalize_il_id(m.group(1))
                if norm:
                    cands.append(norm)
        # 9-digit sequence near ID cue
        for m in re.finditer(r"(?:תעודת\s*זהות|ת\.?\s*ז\.?)[^\n\r]{0,32}?([0-9]{7,10})", text):
            norm = _normalize_il_id(m.group(1))
            if norm:
                cands.append(norm)
        # dedup keep order
        out: List[str] = []
        seen: set[str] = set()
        for v in cands:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _extract_birth_date(text: str) -> Optional[str]:
        for m in re.finditer(r"(?:תאריך\s*לידה|ת\.\s*לידה|לידה)\s*[:\-]?\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", text):
            d, mth, y = m.group(1), m.group(2), m.group(3)
            try:
                yy = int(y)
                if yy < 100:
                    yy += 2000 if yy <= 35 else 1900
                dt = datetime(yy, int(mth), int(d)).date()
                return format_date_for_display(dt)
            except Exception:
                continue
        return None

    def _extract_claim_policy_numbers(text: str) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {"תביעה": [], "פוליסה": []}
        for m in re.finditer(r"תביעה\s*מס'?\s*([0-9]{5,20})", text):
            out["תביעה"].append(m.group(1))
        for m in re.finditer(r"פוליסה\s*מס'?\s*([0-9]{5,20})", text):
            out["פוליסה"].append(m.group(1))
        out["תביעה"] = list(dict.fromkeys(out["תביעה"]))
        out["פוליסה"] = list(dict.fromkeys(out["פוליסה"]))
        return out

    def _extract_medications_from_text(text: str) -> List[str]:
        meds: List[str] = []
        # Common Latin drug names with dosage
        for m in re.finditer(r"\b([A-Z][a-zA-Z]{2,})(?:\s+\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|IU))?\b", text):
            token = m.group(0).strip()
            if len(token) >= 4:
                meds.append(token)
        # Hebrew cues after labels
        for m in re.finditer(r"(?:תרופות|טיפול\s*תרופתי|מרשם)[:\-]?\s*(.+)", text):
            seg = m.group(1)
            for part in re.split(r"[,;\n]", seg):
                pt = part.strip()
                if len(pt) >= 3:
                    meds.append(pt)
        meds = [re.sub(r"\s{2,}", " ", x).strip() for x in meds]
        dedup: List[str] = []
        seen: set[str] = set()
        for m in meds:
            if m not in seen:
                seen.add(m)
                dedup.append(m)
        return dedup[:10]

    # Question routing for the new types
    ql = question_he or ""
    if any(k in ql for k in ["ת.ז", "תז", "תעודת זהות", "מספר זהות", 'ת"ז']):
        page_text_map = _load_page_text_map(ocr_dir)
        page_to_doc = _load_structured_docs_map(ocr_dir)
        for pg, txt in page_text_map.items():
            ids = _extract_il_id_from_text(txt)
            if ids:
                doc = page_to_doc.get(pg, {})
                did = str(doc.get("document_id") or doc.get("original_id") or (rows[0].get("document_id") if rows else ""))
                src = [{
                    "document_id": did,
                    "document_type": doc.get("document_type") if doc else (rows[0].get("document_type") if rows else None),
                    "document_date": doc.get("document_date") if doc else (rows[0].get("document_date") if rows else None),
                    "pages": [pg],
                    "quote": txt[:800],
                }]
                artifacts = _create_response_artifacts(src, ocr_dir)
                return {
                    "question": question_he,
                    "answer": f"מספר תעודת הזהות של המבוטח: {ids[0]}.",
                    "sources": src,
                    "artifacts": artifacts,
                }
        # Not found deterministically
        return {
            "question": question_he,
            "answer": "לא ניתן לקבוע את מספר תעודת הזהות על סמך המסמכים הזמינים.",
            "sources": [],
            "artifacts": _create_response_artifacts([], ocr_dir),
        }

    if any(k in ql for k in ["תאריך לידה", "ת.לידה", 'ת"לידה']):
        page_text_map = _load_page_text_map(ocr_dir)
        page_to_doc = _load_structured_docs_map(ocr_dir)
        for pg, txt in page_text_map.items():
            bd = _extract_birth_date(txt)
            if bd:
                doc = page_to_doc.get(pg, {})
                did = str(doc.get("document_id") or doc.get("original_id") or (rows[0].get("document_id") if rows else ""))
                src = [{
                    "document_id": did,
                    "document_type": doc.get("document_type") if doc else (rows[0].get("document_type") if rows else None),
                    "document_date": doc.get("document_date") if doc else (rows[0].get("document_date") if rows else None),
                    "pages": [pg],
                    "quote": txt[:800],
                }]
                artifacts = _create_response_artifacts(src, ocr_dir)
                return {
                    "question": question_he,
                    "answer": f"תאריך הלידה של המבוטח: {bd}.",
                    "sources": src,
                    "artifacts": artifacts,
                }
        return {
            "question": question_he,
            "answer": "לא צוין תאריך לידה ברור במסמכים הזמינים.",
            "sources": [],
            "artifacts": _create_response_artifacts([], ocr_dir),
        }

    if ("מס" in ql or "מספר" in ql) and ("תביעה" in ql or "פוליסה" in ql or "ביטוח" in ql):
        best_pg = None
        best_text = None
        page_text_map = _load_page_text_map(ocr_dir)
        for pg, txt in page_text_map.items():
            if ("תביעה" in txt) or ("פוליסה" in txt):
                best_pg = pg
                best_text = txt
                break
        if best_text is None and rows:
            best_text = rows[0].get("text") or ""
        nums = _extract_claim_policy_numbers(best_text or "")
        parts: List[str] = []
        if nums.get("תביעה"):
            parts.append(f"מספר התביעה: {', '.join(nums['תביעה'][:3])}")
        if nums.get("פוליסה"):
            parts.append(f"מספר הפוליסה: {', '.join(nums['פוליסה'][:3])}")
        if parts:
            ans = ". ".join(parts) + "."
            page_to_doc = _load_structured_docs_map(ocr_dir)
            doc = page_to_doc.get(best_pg, {}) if best_pg is not None else {}
            did = str(doc.get("document_id") or doc.get("original_id") or (rows[0].get("document_id") if rows else ""))
            src = [{
                "document_id": did,
                "document_type": doc.get("document_type") if doc else (rows[0].get("document_type") if rows else None),
                "document_date": doc.get("document_date") if doc else (rows[0].get("document_date") if rows else None),
                "pages": [best_pg] if best_pg is not None else (rows[0].get("pages") if rows else []),
                "quote": (best_text or "")[:800],
            }]
            artifacts = _create_response_artifacts(src, ocr_dir)
            return {
                "question": question_he,
                "answer": ans,
                "sources": src,
                "artifacts": artifacts,
            }
        return {
            "question": question_he,
            "answer": "לא נמצאו מספרי תביעה או פוליסה במסמכים הזמינים.",
            "sources": [],
            "artifacts": _create_response_artifacts([], ocr_dir),
        }

    if any(k in ql for k in ["תרופות", "טיפול תרופתי", "מרשם"]):
        meds: List[str] = []
        for r in rows[:4]:
            meds.extend(_extract_medications_from_text(r.get("text") or ""))
        if not meds:
            page_text_map = _load_page_text_map(ocr_dir)
            for _, txt in list(page_text_map.items())[:10]:
                meds.extend(_extract_medications_from_text(txt or ""))
        meds = list(dict.fromkeys([m for m in meds if len(m) >= 3]))[:10]
        if meds:
            ans = "תרופות המתועדות במסמכים: " + ", ".join(meds) + "."
            src: List[Dict[str, Any]] = []
            for r in rows[:2]:
                src.append({
                    "document_id": str(r.get("document_id") or ""),
                    "document_type": r.get("document_type"),
                    "document_date": r.get("document_date"),
                    "pages": r.get("pages"),
                    "quote": (r.get("text") or "")[:800],
                })
            artifacts = _create_response_artifacts(src, ocr_dir)
            return {
                "question": question_he,
                "answer": ans,
                "sources": src,
                "artifacts": artifacts,
            }
        return {
            "question": question_he,
            "answer": "לא זוהו תרופות באופן חד-משמעי במסמכים הזמינים.",
            "sources": [],
            "artifacts": _create_response_artifacts([], ocr_dir),
        }

    if "אישור מחלה" in question_he or "חופש מחלה" in question_he or "תוקף" in question_he:
        verdict, sl_sources = _determine_sick_leave_valid(rows)
        if verdict is not None:
            # Create richer answer for sick leave with date and document type
            today = get_jerusalem_date_string()
            if verdict:
                answer_text = f"כן, יש אישור מחלה בתוקף ({today})."
            else:
                answer_text = f"לא, אין אישור מחלה בתוקף ({today})."

            # Add document context if available
            if sl_sources:
                doc_type = sl_sources[0].get("document_type", "")
                doc_date = sl_sources[0].get("document_date", "")
                if doc_date:
                    try:
                        doc_date_formatted = format_date_for_display(datetime.strptime(doc_date, "%Y-%m-%d").date())
                        answer_text = f"{answer_text} {doc_type} מתאריך {doc_date_formatted}."
                    except Exception:
                        pass
                page_text_map = _load_page_text_map(ocr_dir)
                enriched_sources = _enrich_sources_with_page_text(sl_sources[:2], page_text_map)
                artifacts = _create_response_artifacts(enriched_sources, ocr_dir)
                return {
                    "question": question_he,
                    "answer": answer_text,
                    "sources": enriched_sources,
                    "artifacts": artifacts,
                }

    if "מצבו התפקודי" in question_he or "מצב תפקודי" in question_he:
        ans, src = _determine_functional_status(rows)
        if ans:
            print(f"Deterministic functional status answer: {ans[:100]}... ({len(ans)} chars)")
            print("WAITING for coherent enhancement to complete...")
            
            # MUST WAIT for coherent response generation to complete before returning
            try:
                from .coherent_response_generator import create_coherent_medical_response
                context_for_coherence = []
                for r in rows[:3]:
                    context_for_coherence.append({
                        "text": r.get("text", ""),
                        "document_type": r.get("document_type"),
                        "document_date": r.get("document_date")
                    })
                
                print("Starting coherent enhancement...")
                coherent_answer = create_coherent_medical_response(question_he, src, context_for_coherence)
                print(f"Coherent enhancement completed: {len(coherent_answer) if coherent_answer else 0} chars")
                
                if coherent_answer and len(coherent_answer.strip()) >= 30:
                    print(f"USING enhanced coherent answer: {coherent_answer[:100]}...")
                    ans = coherent_answer
                else:
                    print("Coherent answer too short, keeping deterministic")
                    
            except Exception as e:
                print(f"Coherent enhancement failed for functional status: {e}")
                print("Using deterministic answer as fallback")
                
            # Format response with artifacts
            artifacts = _create_response_artifacts(src, ocr_dir)
            final_answer = {
                "question": question_he,
                "answer": ans,
                "sources": src,
                "artifacts": artifacts,
            }
            print(f"RETURNING final functional status answer: {ans[:100]}...")
            return final_answer

    if "רופא התעסוקתי" in question_he or "חזרה לעבודה" in question_he:
        ans, src = _determine_return_to_work(rows)
        if ans:
            print(f"Deterministic return-to-work answer: {ans[:100]}... ({len(ans)} chars)")
            print("WAITING for coherent enhancement to complete...")
            
            # MUST WAIT for coherent response generation to complete before returning
            try:
                from .coherent_response_generator import create_coherent_medical_response
                context_for_coherence = []
                for r in rows[:3]:
                    context_for_coherence.append({
                        "text": r.get("text", ""),
                        "document_type": r.get("document_type"),
                        "document_date": r.get("document_date")
                    })
                
                print("Starting coherent enhancement...")
                coherent_answer = create_coherent_medical_response(question_he, src, context_for_coherence)
                print(f"Coherent enhancement completed: {len(coherent_answer) if coherent_answer else 0} chars")
                
                if coherent_answer and len(coherent_answer.strip()) >= 30:
                    print(f"USING enhanced coherent answer: {coherent_answer[:100]}...")
                    ans = coherent_answer
                else:
                    print("Coherent answer too short, keeping deterministic")
                    
            except Exception as e:
                print(f"Coherent enhancement failed for return-to-work: {e}")
                print("Using deterministic answer as fallback")
                
            artifacts = _create_response_artifacts(src, ocr_dir)
            final_answer = {
                "question": question_he,
                "answer": ans,
                "sources": src,
                "artifacts": artifacts,
            }
            print(f"RETURNING final return-to-work answer: {ans[:100]}...")
            return final_answer

    # If none of the deterministic handlers matched, call the LLM with cleaned context
    try:
        context = build_context_rows(rows)
        prompt = prompt_for_question(question_he, context)
        out = gemini_generate(prompt, system=SYSTEM_PROMPT_HE, model="gemini-1.5-flash", temperature=0.2, json_only=True)
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to ollama
        try:
            out = ollama_generate(prompt, model="gemma2:2b-instruct", system=SYSTEM_PROMPT_HE, json_only=True)
        except Exception:
            out = {}

    # Build sources deterministically from retrieval (ignore model-provided sources)
    seen: set[str] = set()
    sources: List[Dict[str, Any]] = []
    for r in rows:
        did = str(r.get("document_id") or "")
        if not did or did in seen:
            continue
        seen.add(did)
        quote = (r.get("text") or "").strip()
        sources.append({
            "document_id": did,
            "document_type": r.get("document_type"),
            "document_date": r.get("document_date"),
            "pages": r.get("pages"),
            "quote": quote,
        })
        if len(sources) >= 2:
            break

    # Extract answer text from model or fallback
    answer_text = None
    if isinstance(out, dict):
        answer_text = out.get("answer")
    if not isinstance(answer_text, str) or not answer_text.strip():
        answer_text = "לא נמצאה תשובה חד-משמעית על סמך המקורות שסופקו."
    
    print(f"Initial answer from model: {answer_text[:100]}... ({len(answer_text)} chars)")
    print(f"Answer contains 'ממצאים': {answer_text.count('ממצאים')} times")

    # Always apply refinement for better coherence
    should_refine = _needs_refine(answer_text, rows) or len(answer_text) > 300 or answer_text.count("ממצאים") > 1
    if should_refine:
        try:
            # Build a concise Hebrew bullet gist from sources only
            context_for_refine = []
            for r in rows[:4]:
                txt = _clean_context_text(r.get("text") or "")
                if not txt:
                    continue
                # Prefer the most recent docs
                context_for_refine.append({
                    "document_type": r.get("document_type"),
                    "date": r.get("document_date"),
                    "snippet": txt[:1200]
                })
            refine_prompt = (
                "צור תשובה מקיפה, ברורה וקוהרנטית בעברית על בסיס המידע הרפואי שלפניך.\n"
                "התשובה חייבת להיות:\n"
                "• כתובה במשפטים מלאים וברורים\n"
                "• מאורגנת באופן לוגי וקל להבנה\n"
                "• כוללת את כל המידע הרלוונטי מהמקורות\n"
                "• ללא משפטים חתוכים או לא גמורים\n"
                "• מובנת למי שקורא אותה לראשונה\n\n"
                f"שאלה שנשאלה: {question_he}\n\n"
                f"טיוטת תשובה לשיפור: {answer_text}\n\n"
                f"מקורות רפואיים זמינים:\n{json.dumps(context_for_refine, ensure_ascii=False, indent=2)}\n\n"
                "צור תשובה משופרת וקוהרנטית שעונה על השאלה בצורה מלאה.\n"
                "החזר JSON בפורמט: {\"answer\": \"התשובה המלאה והקוהרנטית כאן\"}"
            )
            refined = gemini_generate(refine_prompt, system="אתה רופא מומחה המתמחה ביצירת תשובות רפואיות מקיפות, ברורות וקוהרנטיות בעברית. המטרה שלך היא לקחת מידע רפואי גולמי ולהפוך אותו לתשובה מובנת ושלמה. שמור על דיוק עובדתי מוחלט ותכתוב רק במשפטים מלאים וברורים.", model="gemini-1.5-flash", temperature=0.2, json_only=True)
            if isinstance(refined, dict) and isinstance(refined.get("answer"), str) and refined.get("answer").strip():
                answer_text = refined["answer"].strip()
                print(f"Refined answer: {answer_text[:100]}... ({len(answer_text)} chars)")
        except Exception as e:
            print(f"Refinement failed: {e}")
    
    # Apply coherent response generation for all answers
    try:
        from .coherent_response_generator import create_coherent_medical_response
        print(f"Applying coherent generation to answer: {len(answer_text)} chars")
        
        context_for_coherence = []
        for r in rows[:3]:
            context_for_coherence.append({
                "text": r.get("text", ""),
                "document_type": r.get("document_type"),
                "document_date": r.get("document_date")
            })
        
        coherent_answer = create_coherent_medical_response(question_he, sources, context_for_coherence)
        if coherent_answer and len(coherent_answer.strip()) >= 30:
            print(f"Successfully generated coherent answer: {len(coherent_answer)} chars")
            answer_text = coherent_answer
        else:
            print(f"Coherent answer too short, keeping original")
    except Exception as e:
        print(f"Coherent response generation failed: {e}")
        import traceback
        traceback.print_exc()

    latin_ratio = sum(1 for c in answer_text if 'a' <= c.lower() <= 'z') / max(1, len(answer_text))
    if latin_ratio > 0.2:
        answer_text = "לא נמצאה תשובה חד-משמעית על סמך המקורות שסופקו."

    # Always use coherent response generator for better quality
    try:
        from .coherent_response_generator import create_coherent_medical_response
        print(f"Original answer length: {len(answer_text)} chars")
        
        context_for_coherence = []
        for r in rows[:3]:
            context_for_coherence.append({
                "text": r.get("text", ""),
                "document_type": r.get("document_type"),
                "document_date": r.get("document_date")
            })
        
        coherent_answer = create_coherent_medical_response(question_he, sources, context_for_coherence)
        if coherent_answer and len(coherent_answer.strip()) >= 30:
            print(f"Using coherent answer: {len(coherent_answer)} chars")
            answer_text = coherent_answer
        else:
            print(f"Coherent answer too short: {len(coherent_answer) if coherent_answer else 0} chars")
    except Exception as e:
        print(f"Coherent response generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Ensure answer ends properly
    answer_text = answer_text.strip()
    if answer_text and not answer_text.endswith(('.', '!', '?')):
        answer_text += '.'

    # Add artifacts to the response
    # Enrich sources' quotes from aligned page text so it reflects updated canonical content
    page_text_map = _load_page_text_map(ocr_dir)
    sources = _enrich_sources_with_page_text(sources, page_text_map)
    artifacts = _create_response_artifacts(sources, ocr_dir)
    return {
        "question": question_he,
        "answer": answer_text,
        "sources": sources,
        "artifacts": artifacts,
    }


def _refine_answer(
    question_he: str,
    draft_answer: str,
    sources: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    model: str,
) -> str:
    """Short CoT rewriter: 2–4 coherent Hebrew sentences grounded in sources.
    Returns the refined answer or the draft if refinement fails.
    """
    try:
        doc_map: Dict[str, str] = {}
        for r in rows[:8]:
            did = str(r.get("document_id") or "")
            if did and did not in doc_map:
                doc_map[did] = _clean_context_text(r.get("text") or "")[:1200]
        ctx: List[Dict[str, Any]] = []
        for s in sources[:3]:
            did = str(s.get("document_id") or "")
            ctx.append({
                "document_type": s.get("document_type"),
                "date": s.get("document_date"),
                "snippet": (s.get("quote") or "")[:800] or doc_map.get(did, "")[:800],
            })
        refine_prompt = (
            "שפר את ניסוח התשובה באופן מפורט, קוהרנטי ומלא בעברית.\n"
            "ודא שכל המידע הרלוונטי נכלל ושאין משפטים חתוכים באמצע.\n"
            "הסתמך אך ורק על המידע הבא מהמסמכים; אין להוסיף מידע.\n\n"
            f"שאלה: {question_he}\n\n"
            f"טיוטה: {draft_answer}\n\n"
            f"מקורות (JSON):\n{json.dumps(ctx, ensure_ascii=False)}\n\n"
            "החזר JSON יחיד במבנה: {\"answer\": \"...\"}."
        )
        refined = gemini_generate(
            refine_prompt,
            system="אתה רופא מומחה הנותח ניסוח של תשובות רפואיות בעברית. נסח באופן מפורט, מדויק וקוהרנטי על בסיס המידע שסופק בלבד. ודא שכל המידע הרלוונטי נכלל.",
            model="gemini-1.5-flash",
            temperature=0.2,
            max_tokens=1024,
            json_only=True,
        )
        if isinstance(refined, dict) and isinstance(refined.get("answer"), str):
            out = refined.get("answer", "").strip()
            if out:
                return out
    except Exception:
        pass
    return draft_answer


def _create_response_artifacts(sources: List[Dict[str, Any]], ocr_dir: str) -> Dict[str, Any]:
    """Create accordion artifacts with source PDFs and relevant pages.

    Primary source: results_aligned.json (preferred) -> pages[].
    Fallbacks:
      - results_aligned_cleaned.json (if present)
      - results.json
      - structured_documents.jsonl (per-document text used for page content)
      - full_text_aligned.txt (split into page-like chunks if possible)
    """
    import os
    import json

    artifacts = {
        "pdfs": [],
        "pages": []
    }

    # Load results_aligned.json to get page data
    page_data = []
    def _try_load_json(path: str) -> list:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("pages", []) if isinstance(data, dict) else []
        except Exception:
            return []

    # Try preferred cleaned/aligned file
    page_data = _try_load_json(os.path.join(ocr_dir, "results_aligned_cleaned.json"))
    if not page_data:
        page_data = _try_load_json(os.path.join(ocr_dir, "results_aligned.json"))
    if not page_data:
        # Fallback legacy variant name
        page_data = _try_load_json(os.path.join(ocr_dir, "results_aligned_clean.json"))
    if not page_data:
        # Try original results.json
        page_data = _try_load_json(os.path.join(ocr_dir, "results.json"))

    # Add PDF artifact - check both ocr_dir and parent directory
    pdf_path = os.path.join(ocr_dir, "med_patient#1.pdf")
    if not os.path.exists(pdf_path):
        # Try parent directory
        parent_dir = os.path.dirname(ocr_dir)
        pdf_path = os.path.join(parent_dir, "med_patient#1.pdf")

    if os.path.exists(pdf_path):
        artifacts["pdfs"].append({
            "name": "מסמך רפואי מקורי",
            "path": pdf_path,
            "type": "pdf"
        })

    # Process each source to get relevant pages
    relevant_pages = set()
    for source in sources:
        pages = source.get("pages", [])
        if isinstance(pages, list):
            relevant_pages.update(pages)

    # 1) If we have page_data from aligned/original results, use it
    if page_data:
        for page_num in sorted(relevant_pages):
            if 1 <= page_num <= len(page_data):
                page_info = page_data[page_num - 1]  # 0-indexed
                # Prefer full text. If "text" looks truncated, rebuild from lines
                content_text = page_info.get("text", "") or ""
                if (not content_text.strip() or len(content_text.strip()) < 200) and isinstance(page_info.get("lines"), list):
                    try:
                        content_text = "\n".join(line.get("text", "") for line in page_info.get("lines") if isinstance(line, dict))
                    except Exception:
                        pass
                artifacts["pages"].append({
                    "page_number": page_num,
                    "content": content_text,
                    "metadata": {
                        "width": page_info.get("width", 0),
                        "height": page_info.get("height", 0),
                        "lines_count": len(page_info.get("lines", []))
                    }
                })

    # 2) Fallback: use structured_documents.jsonl text if we couldn't pull any page content
    if not artifacts["pages"]:
        try:
            struct_path = os.path.join(ocr_dir, "structured_documents.jsonl")
            doc_map = {}
            with open(struct_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc_map[str(row.get("document_id") or "")] = row
            # Build a page artifact per referenced page using the document's text as content (best-effort)
            for source in sources:
                did = str(source.get("document_id") or "")
                doc = doc_map.get(did)
                if not doc:
                    continue
                doc_text = (doc.get("text") or "").strip()
                for page_num in (source.get("pages") or []):
                    artifacts["pages"].append({
                        "page_number": int(page_num),
                        "content": doc_text[:5000],
                        "metadata": {"width": 0, "height": 0, "lines_count": doc_text.count('\n') + 1}
                    })
        except Exception:
            pass

    return artifacts
