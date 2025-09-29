import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PageMeta:
    page_number: int
    text: str
    width: int
    height: int
    raw_lines: List[Dict[str, Any]]
    # Heuristics detected on the page
    document_type: Optional[str] = None
    document_date: Optional[str] = None
    issuer: Optional[str] = None
    document_id: Optional[str] = None
    page_of_total: Optional[Tuple[int, int]] = None
    patient_name: Optional[str] = None
    question_hints: List[str] = field(default_factory=list)


@dataclass
class DocumentRecord:
    # Non-default fields first
    document_id: str
    original_id: str
    document_type: str
    document_date: Optional[str]
    issuer: Optional[str]
    pages: List[int]
    text: str
    source: str
    patient_name: Optional[str]
    # Defaulted fields follow
    found_dates: List[str] = field(default_factory=list)
    detected_keywords: List[str] = field(default_factory=list)
    # Normalization and date orientation
    category: str = "אחר"  # ביקור רופא כללי | ביקור רופא תעסוקתי | ביקור במיון | טופס ביטוח | אחר
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    sort_date: Optional[str] = None
    document_date_il: Optional[str] = None
    date_start_il: Optional[str] = None
    date_end_il: Optional[str] = None
    sort_date_il: Optional[str] = None
    chron_id: int = 0


# -----------------
# Regex utilities
# -----------------

_DATE_RE = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")
_PAGE_OF_TOTAL_RE = re.compile(r"(?:(?:דף|עמוד)\s*(?:מס׳|מסי|מס|#)?\s*)?(\d{1,3})\s*/\s*(\d{1,3})")


def _is_top_region(y: float, page_height: int, fraction: float = 0.25) -> bool:
    try:
        return float(y) <= float(page_height) * fraction
    except Exception:
        return False


def _has_doc_header(raw_lines: List[Dict[str, Any]], page_height: int) -> bool:
    if not raw_lines:
        return False
    header_pats: List[str] = [
        r"דוח\s*(?:שחרור|ביקור\s*מרפאה|מיון)",
        r"סיכום\s*(?:טיפול|בדיקה)",
        r"רפואה\s*תעסוקתית|רופא\s*תעסוקתי",
        r"תעודה\s*רפואית|אישור\s*מחלה",
        r"בל\/?250|טיפול רפואי לנפגע עבודה",
        r"הזמנת\s*בדיקה\s*רפואית",
        r"ועדה\s*רפואית",
        r"HOLY\s+FAMILY\s+HOSPITAL|כללית|המוסד\s+לביטוח\s+לאומי|משרד\s+הפנים|כלל\s+חברה\s+לביטוח|הנדון:\s|פוליסה\s*מס'|תביעה\s*מס'",
        r"דף\s*מס[׳'\-:]?\s*\d+\s*/\s*\d+",
    ]
    for line in raw_lines:
        t = (line.get("text") or "").strip()
        if not t:
            continue
        bbox = line.get("bbox") or [0, 0, 0, 0]
        y1 = bbox[1] if isinstance(bbox, (list, tuple)) and len(bbox) >= 2 else 0
        if not _is_top_region(y1, page_height):
            continue
        for pat in header_pats:
            if re.search(pat, t):
                return True
    return False


def _normalize_date_to_iso(date_text: str) -> Optional[str]:
    try:
        parts = re.split(r"[./-]", date_text)
        if len(parts) != 3:
            return None
        day, month, year = parts
        if len(year) == 2:
            # assume 20xx for 00–79 and 19xx otherwise
            y = int(year)
            year = f"20{y:02d}" if y <= 79 else f"19{y:02d}"
        dt = datetime(int(year), int(month), int(day))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _iso_to_il(date_iso: Optional[str]) -> Optional[str]:
    if not date_iso:
        return None
    try:
        dt = datetime.strptime(date_iso, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return None


def _extract_dates(text: str) -> List[str]:
    dates = []
    for m in _DATE_RE.finditer(text):
        iso = _normalize_date_to_iso(m.group(1))
        if iso:
            dates.append(iso)
    # return unique preserving order, newest last
    seen = set()
    unique = []
    for d in dates:
        if d not in seen:
            unique.append(d)
            seen.add(d)
    return unique


def _extract_page_of_total(text: str) -> Optional[Tuple[int, int]]:
    m = _PAGE_OF_TOTAL_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def _detect_document_type_text_only(text: str) -> Optional[str]:
    patterns: List[Tuple[str, str]] = [
        (r"דוח שחרור מחדר מיון|שחרור מחדר מיון|דוח מיון", "דוח שחרור מחדר מיון"),
        (r"דוח שחרור\b|סיכום אשפוז", "דוח שחרור"),
        (r"דוח ביקור מרפאה|ביקורת מרפאה|סיכום ביקור", "דוח ביקור מרפאה"),
        (r"סיכום טיפול\b|פיזיותרפיה|פיזיוטרפיה", "סיכום טיפול פיזיותרפיה"),
        (r"טופס סיכום בדיקה רפואית|רפואה תעסוקתית|רופא תעסוקתי", "טופס סיכום בדיקה רפואית (רפואה תעסוקתית)"),
        (r"תעודה רפואית (?:ראשונה|נוספת)? לנפגע בעבודה|אישור מחלה\b|מחלה בתוקף", "תעודה רפואית לנפגע בעבודה"),
        (r"החלטת הוועדה הרפואית|דוח ועדה רפואית", "ועדה רפואית"),
        (r"טופס למתן טיפול רפואי לנפגע בעבודה.*\(בל\s*250\)|בל\/?250|בקשה למתן טיפול רפואי לנפגע עבודה", "בל/250"),
        (r"הזמנת בדיקה רפואית", "הזמנת בדיקה רפואית"),
        (r"תעודת זהות|משרד הפנים", "תעודת זהות"),
        (r"כלל\s+חברה\s+לביטוח|מערך\s+התביעות|הנדון:\s|פוליסה\s*מס'|תביעה\s*מס'", "מכתב לחברת ביטוח"),
    ]
    for pat, label in patterns:
        if re.search(pat, text):
            return label
    return None


def _infer_document_type(text: str, raw_lines: List[Dict[str, Any]], page_height: int) -> Optional[str]:
    # Layout-aware scoring: boost matches in top region
    candidates: Dict[str, int] = {}
    layout_patterns: List[Tuple[List[str], str, int]] = [
        ([r"דוח שחרור מחדר מיון", r"שחרור מחדר מיון", r"דוח מיון"], "דוח שחרור מחדר מיון", 3),
        ([r"דוח שחרור\b", r"סיכום אשפוז"], "דוח שחרור", 2),
        ([r"דוח ביקור מרפאה", r"ביקורת מרפאה", r"סיכום ביקור"], "דוח ביקור מרפאה", 2),
        ([r"סיכום טיפול", r"פיזי[ו]?תרפיה"], "סיכום טיפול פיזיותרפיה", 2),
        ([r"רפואה\s*תעסוקתית", r"רופא\s*תעסוקתי", r"טופס סיכום בדיקה רפואית"], "טופס סיכום בדיקה רפואית (רפואה תעסוקתית)", 3),
        ([r"תעודה רפואית", r"אישור מחלה"], "תעודה רפואית לנפגע בעבודה", 3),
        ([r"בל\/?250", r"טיפול רפואי לנפגע בעבודה"], "בל/250", 3),
        ([r"ועדה רפואית"], "ועדה רפואית", 2),
        ([r"הזמנת בדיקה רפואית"], "הזמנת בדיקה רפואית", 2),
        ([r"תעודת זהות|משרד הפנים"], "תעודת זהות", 2),
    ]
    # Pass 1: lines in top region
    for line in raw_lines or []:
        lt = (line.get("text") or "").strip()
        bbox = line.get("bbox") or [0, 0, 0, 0]
        y1 = bbox[1] if isinstance(bbox, (list, tuple)) and len(bbox) >= 2 else 0
        is_top = _is_top_region(y1, page_height)
        if not lt:
            continue
        for pats, label, base in layout_patterns:
            for pat in pats:
                if re.search(pat, lt):
                    score = base + (2 if is_top else 0)
                    candidates[label] = candidates.get(label, 0) + score
                    break
    # Pass 2: full text as fallback
    text_label = _detect_document_type_text_only(text)
    if text_label:
        candidates[text_label] = candidates.get(text_label, 0) + 1
    if not candidates:
        return None
    # Choose max score
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def _extract_document_id(text: str) -> Optional[str]:
    # Look for typical ids
    candidates: List[str] = []
    # EM / ME codes
    candidates += re.findall(r"\b(?:EM|ME)\d{6,}\b", text)
    # Numeric ids after common headers (robust to OCR noise around 'דוח'/'ביקור')
    for m in re.finditer(r"(?:ד[וחוה]|דו[חח])\s*(?:שחרור|ביקור\s*מרפאה)\s*(\d{6,})", text):
        candidates.append(m.group(1))
    # Claim/policy/event numbers on insurance forms
    for m in re.finditer(r"מס(?:פר)?\s*(?:תביעה|פוליסה|ארוע|אירוע|דוח|ביקור)\s*:?\s*(\d{6,})", text):
        candidates.append(m.group(1))
    # If multiple, pick the longest then first
    candidates = sorted(set(candidates), key=lambda s: (-len(s), s))
    if candidates:
        return candidates[0]
    return None


def _extract_visit_report_id(text: str) -> Optional[str]:
    # IDs like: "דוח ביקור מרפאה 45062778" (OCR may mangle 'דוח')
    m = re.search(r"ביקור\s*מרפאה\s*(\d{6,})", text)
    if m:
        return m.group(1)
    return None


def _detect_issuer(text: str) -> Optional[str]:
    issuer_patterns: List[Tuple[str, str]] = [
        (r"HOLY\s+FAMILY\s+HOSPITAL|משפחה קדושה", "HOLY FAMILY HOSPITAL"),
        (r"כללית|שירותי בריאות כללית", "כללית"),
        (r"המוסד לביטוח לאומי|הביטוח הלאומי", "המוסד לביטוח לאומי"),
        (r"משרד הפנים", "משרד הפנים"),
        (r"ד" + r"""'""" + r"ר\s+[\u0590-\u05FF\s\-']{2,}", "רופא/ה")
    ]
    for pat, label in issuer_patterns:
        if re.search(pat, text):
            return label
    return None


def _detect_patient_name(text: str) -> Optional[str]:
    # Heuristic: look for patterns like "משפחה ושם:" or just line containing two tokens followed by digits id nearby
    m = re.search(r"משפחה ושם\s*[:：]\s*([\u0590-\u05FF\s']{2,})", text)
    if m:
        name = m.group(1).strip()
        name = re.sub(r"\s+", " ", name)
        return name
    m = re.search(r"שם\s*(?:משפחה\s*)?ושם\s*פרטי\s*\n?([\u0590-\u05FF\s']{2,})", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(r"שם\s*המבוטח\s*[:：]\s*([\u0590-\u05FF\s']{2,})", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def _detect_question_hints(text: str) -> List[str]:
    hints: List[str] = []
    if re.search(r"אישור\s*מחלה|חופש\s*מחלה|תוקף\s*המגבלה|עד\s*תאריך", text):
        hints.append("אישור/חופש מחלה")
    if re.search(r"רפואה\s*תעסוקתית|כושר\s*עבודה|אינו\s*יכול\s*לחזור|תעסוקתית", text):
        hints.append("רפואה תעסוקתית")
    if re.search(r"כאבים|מגבלה|טווחי\s*תנועה|אגרוף|רגישות\s*במישוש|NV", text):
        hints.append("מצב תפקודי")
    return hints


def _normalize_category(doc_type: Optional[str], text: str) -> str:
    t = doc_type or ""
    # Direct mapping by detected document_type
    if t in {"דוח שחרור מחדר מיון", "דוח מיון"}:
        return "ביקור במיון"
    if t in {"דוח שחרור", "דוח ביקור מרפאה"}:
        return "ביקור רופא כללי"
    if "רפואה תעסוקתית" in t:
        return "ביקור רופא תעסוקתי"
    if t in {"תעודה רפואית לנפגע בעבודה", "בל/250", "הזמנת בדיקה רפואית", "ועדה רפואית"}:
        return "טופס ביטוח"
    if t in {"סיכום טיפול פיזיותרפיה", "תעודת זהות"}:
        return "אחר"

    # Heuristics on raw text when type is UNKNOWN
    tx = text
    if re.search(r"מיון|חדר\s*מיון|EM\d{5,}", tx):
        return "ביקור במיון"
    if re.search(r"רפואה\s*תעסוקתית|כושר\s*עבודה", tx):
        return "ביקור רופא תעסוקתי"
    if re.search(r"ביקור\s*מרפאה|מרפאת|אורתופדית|שחרור", tx):
        return "ביקור רופא כללי"
    if re.search(r"פוליסה|חברת\s*ביטוח|תביעה|ביטוח\s*לאומי|ועדה\s*רפואית|בל\/?250|נספח", tx):
        return "טופס ביטוח"
    return "אחר"


def _find_labeled_dates(text: str, raw_lines: List[Dict[str, Any]]) -> List[Tuple[str, str, int, float]]:
    """
    Return list of (iso_date, label, weight, y_position). Labels include:
    "קבלה", "שחרור", "ביקור", "בדיקה", "מתאריך", "עד", "תוקף", "הנפקה", "כללי".
    Weights are boosted for dates appearing on lines with the label and in top region.
    """
    results: List[Tuple[str, str, int, float]] = []
    # Line-based labeled extraction
    label_specs: List[Tuple[str, List[str]]] = [
        ("שחרור", [r"תאריך\s*שחרור", r"שחרור:\s*"]),
        ("קבלה", [r"תאריך\s*קבלה", r"קבלה:\s*"]),
        ("ביקור", [r"תאריך\s*ביקור", r"ביקור:\s*"]),
        ("בדיקה", [r"תאריך\s*בדיקה", r"בדיקה:\s*"]),
        ("כניסה", [r"תאריך\s*כניסה", r"כניסה:\s*"]),
        ("שחרור מרפאה", [r"תאריך\s*שח\.?\s*מרפאה", r"שח\.?\s*מרפאה:\s*"]),
        ("מתאריך", [r"מתאריך", r"מ-\s*"]),
        ("עד", [r"עד\s*תאריך", r"עד\s*", r"תוקף עד"]),
        ("תוקף", [r"תוקף\s*(?:עד)?"]),
        ("הנפקה", [r"תאריך\s*הנפקה", r"הונפק ביום|הונפק בתאריך"]),
        ("כללי", [r"תאריך\b", r"Date\b"]),
    ]
    exclude_line = re.compile(r"לידה|תאריך\s*לידה|הודפס|הדפסת|שעת\s*הדפסת|Printed")
    for line in raw_lines or []:
        lt = (line.get("text") or "").strip()
        bbox = line.get("bbox") or [0, 0, 0, 0]
        y1 = bbox[1] if isinstance(bbox, (list, tuple)) and len(bbox) >= 2 else 0.0
        if not lt:
            continue
        if exclude_line.search(lt):
            continue
        for label, pats in label_specs:
            if any(re.search(p, lt) for p in pats):
                for m in _DATE_RE.finditer(lt):
                    iso = _normalize_date_to_iso(m.group(1))
                    if iso:
                        base = 3
                        if _is_top_region(y1, int(bbox[3] if len(bbox) >= 4 else (y1 + 1))):
                            base += 1
                        results.append((iso, label, base, float(y1)))
                break
    # Fallback: context in full text around keywords
    context_patterns: List[Tuple[str, str]] = [
        (r"תאריך\s*שחרור\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "שחרור"),
        (r"תאריך\s*קבלה\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "קבלה"),
        (r"תאריך\s*ביקור\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "ביקור"),
        (r"תאריך\s*בדיקה\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "בדיקה"),
        (r"תאריך\s*כניסה\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "כניסה"),
        (r"תאריך\s*שח\.?\s*מרפאה\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "שחרור מרפאה"),
        (r"מתאריך\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "מתאריך"),
        (r"עד\s*תאריך\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "עד"),
        (r"תוקף(?:\s*עד)?\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "תוקף"),
        (r"תאריך\s*הנפקה\D*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", "הנפקה"),
    ]
    for pat, label in context_patterns:
        m = re.search(pat, text)
        if m:
            iso = _normalize_date_to_iso(m.group(1))
            if iso:
                results.append((iso, label, 2, 0.0))
    # Generic dates as low-priority (line-based, excluding birth/printed lines)
    for line in raw_lines or []:
        lt = (line.get("text") or "").strip()
        bbox = line.get("bbox") or [0, 0, 0, 0]
        y1 = bbox[1] if isinstance(bbox, (list, tuple)) and len(bbox) >= 2 else 0.0
        if not lt or exclude_line.search(lt):
            continue
        for m in _DATE_RE.finditer(lt):
            iso = _normalize_date_to_iso(m.group(1))
            if iso:
                results.append((iso, "כללי", 1, float(y1)))
    return results


def _choose_document_date_for_type(doc_type: Optional[str], labeled_dates: List[Tuple[str, str, int, float]], fallback_dates: List[str]) -> Optional[str]:
    if not labeled_dates and not fallback_dates:
        return None
    # Preference order by type
    preferences: Dict[str, Tuple[List[str], str]] = {
        # label order, selection strategy for label (latest | top)
        "דוח שחרור מחדר מיון": (["שחרור", "קבלה", "ביקור", "כללי"], "latest"),
        "דוח שחרור": (["שחרור", "קבלה", "כללי"], "latest"),
        "דוח ביקור מרפאה": (["ביקור", "כניסה", "שחרור מרפאה", "בדיקה", "כללי"], "latest"),
        "טופס סיכום בדיקה רפואית (רפואה תעסוקתית)": (["בדיקה", "ביקור", "כללי"], "latest"),
        "סיכום טיפול פיזיותרפיה": (["ביקור", "בדיקה", "כללי"], "latest"),
        "תעודה רפואית לנפגע בעבודה": (["הנפקה", "בדיקה", "ביקור", "כללי"], "top"),
        "בל/250": (["הנפקה", "כללי"], "top"),
        "הזמנת בדיקה רפואית": (["הנפקה", "כללי"], "top"),
        "מכתב לחברת ביטוח": (["כללי", "הנפקה", "בדיקה"], "top"),
    }
    order, strategy = preferences.get(doc_type or "", (["שחרור", "בדיקה", "ביקור", "קבלה", "מתאריך", "עד", "כללי"], "latest"))

    # Group entries by label
    grouped: Dict[str, List[Tuple[str, int, float]]] = {}
    for iso, label, weight, y in labeled_dates:
        grouped.setdefault(label, []).append((iso, weight, y))

    def pick(entries: List[Tuple[str, int, float]], mode: str) -> str:
        if mode == "top":
            # Prefer top-region (smallest y), break ties by highest weight, then latest date
            return sorted(entries, key=lambda t: (t[2], -t[1], t[0]))[0][0]
        # latest
        return sorted(entries, key=lambda t: (t[0], -t[1], t[2]))[-1][0]

    for lbl in order:
        if lbl in grouped and grouped[lbl]:
            return pick(grouped[lbl], "top" if lbl == "כללי" or strategy == "top" else "latest")

    # Fallback: if we saw generic dates (filter out implausible birth/old dates)
    if fallback_dates:
        plausible = [d for d in fallback_dates if isinstance(d, str) and d >= "2000-01-01"]
        pool = plausible or fallback_dates
        try:
            # Prefer earliest in text (often header)
            return sorted(set(pool))[0]
        except Exception:
            return pool[0]
    return None


def _analyze_page(page: Dict[str, Any]) -> PageMeta:
    text: str = page.get("text", "") or ""
    page_number: int = int(page.get("page", 0))
    width: int = int(page.get("width", 0))
    height: int = int(page.get("height", 0))
    raw_lines = page.get("lines", []) or []

    dates = _extract_dates(text)
    page_of_total = _extract_page_of_total(text)
    doc_type = _infer_document_type(text, raw_lines, height) or _detect_document_type_text_only(text)
    doc_id = _extract_document_id(text)
    issuer = _detect_issuer(text)
    patient = _detect_patient_name(text)
    hints = _detect_question_hints(text)

    return PageMeta(
        page_number=page_number,
        text=text,
        width=width,
        height=height,
        raw_lines=raw_lines,
        document_type=doc_type,
        document_date=None,  # filled below using labeled preferences
        issuer=issuer,
        document_id=doc_id,
        page_of_total=page_of_total,
        patient_name=patient,
        question_hints=hints,
    )


def _make_document_key(pm: PageMeta) -> str:
    # Deprecated: kept for reference, not used in sequential grouping anymore
    type_part = pm.document_type or "UNKNOWN"
    id_part = pm.document_id or f"AUTO-{type_part}"
    issuer_part = pm.issuer or ""
    return "|".join([type_part, id_part, issuer_part])


def _choose_document_date(acc_dates: List[str]) -> Optional[str]:
    if not acc_dates:
        return None
    try:
        # choose the latest date
        acc_dates_sorted = sorted(set(acc_dates))
        return acc_dates_sorted[-1]
    except Exception:
        return acc_dates[-1]


def build_ocr_documents(results_json_path: str, out_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Read OCR results_aligned.json (or fallback to results.json) and produce two files before chunking:
      - structured_documents.jsonl : one line per logical document (joined pages)
      - documents_index.json       : index mapping and quick lookup by document_id

    Returns (jsonl_path, index_path)
    """
    # Prefer cleaned if present, then aligned, then base results.json
    cleaned_path = results_json_path.replace("results.json", "results_aligned_cleaned.json")
    aligned_path = results_json_path.replace("results.json", "results_aligned.json")
    if os.path.isfile(cleaned_path):
        results_json_path = cleaned_path
    elif os.path.isfile(aligned_path):
        results_json_path = aligned_path
    elif not os.path.isfile(results_json_path):
        raise FileNotFoundError(f"Neither {aligned_path} nor {results_json_path} found")

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_pdf: str = data.get("source") or ""
    pages: List[Dict[str, Any]] = data.get("pages", [])
    analyzed: List[PageMeta] = []
    for p in pages:
        pm = _analyze_page(p)
        analyzed.append(pm)

    # Sequential grouping by page-of-total and header signals
    analyzed.sort(key=lambda x: x.page_number)
    documents: List[DocumentRecord] = []
    current: List[PageMeta] = []
    current_total: Optional[int] = None
    next_expected_idx: Optional[int] = None

    def flush_current() -> None:
        nonlocal documents, current
        if not current:
            return
        # Aggregate
        texts = [p.text for p in current if p.text]
        full_text = "\n\n".join(texts)
        page_numbers = [p.page_number for p in current]

        # Doc type: prefer majority across pages; tie-breaker is first page
        types = [p.document_type for p in current if p.document_type]
        doc_type = None
        if types:
            # Count occurrences
            counts: Dict[str, int] = {}
            for t in types:
                counts[t] = counts.get(t, 0) + 1
            max_count = max(counts.values())
            top_labels = [t for t, c in counts.items() if c == max_count]
            if len(top_labels) == 1:
                doc_type = top_labels[0]
            else:
                # tie: use first page's type
                doc_type = types[0]
        doc_type = doc_type or "UNKNOWN"

        # Issuer: prefer first non-empty
        issuer = next((p.issuer for p in current if p.issuer), None)
        # Patient name: first non-empty
        patient = next((p.patient_name for p in current if p.patient_name), None)

        # Dates: labeled across all pages
        labeled_all: List[Tuple[str, str, int, float]] = []
        fallback_dates: List[str] = []
        for p in current:
            labeled_all.extend(_find_labeled_dates(p.text, p.raw_lines))
            fallback_dates.extend(_extract_dates(p.text))
        chosen_date = _choose_document_date_for_type(doc_type, labeled_all, fallback_dates)

        # Document ID: collect candidates
        id_candidates: List[str] = []
        for p in current:
            vid = _extract_visit_report_id(p.text)
            if vid:
                id_candidates.append(vid)
            did = _extract_document_id(p.text)
            if did:
                id_candidates.append(did)
        id_candidates = sorted(set(id_candidates), key=lambda s: (-len(s), s))
        doc_id = id_candidates[0] if id_candidates else f"AUTO-{doc_type}"

        # Accumulate hints and dates set
        dates_set = sorted(set(_normalize_date_to_iso(d) if _normalize_date_to_iso(d) else d for d in fallback_dates if d))
        hints: List[str] = []
        for p in current:
            hints.extend(p.question_hints)

        # Category
        category = _normalize_category(doc_type, full_text)
        date_start = min([d for d in dates_set if d], default=chosen_date)
        date_end = max([d for d in dates_set if d], default=chosen_date)

        doc = DocumentRecord(
            document_id=str(doc_id),
            original_id=str(doc_id),
            document_type=str(doc_type),
            document_date=chosen_date,
            document_date_il=_iso_to_il(chosen_date),
            issuer=issuer,
            pages=page_numbers,
            text=full_text,
            source=source_pdf,
            patient_name=patient,
            found_dates=dates_set,
            detected_keywords=sorted(set(hints)),
            category=category,
            date_start=date_start,
            date_end=date_end,
            sort_date=chosen_date,
            date_start_il=_iso_to_il(date_start),
            date_end_il=_iso_to_il(date_end),
            sort_date_il=_iso_to_il(chosen_date),
        )
        documents.append(doc)

    for pm in analyzed:
        boundary = False
        if not current:
            boundary = True
        else:
            if pm.page_of_total:
                idx, total = pm.page_of_total
                if idx == 1:
                    boundary = True
                elif current_total is not None and total != current_total:
                    boundary = True
                elif next_expected_idx is not None and idx != next_expected_idx:
                    boundary = True
            else:
                if _has_doc_header(pm.raw_lines, pm.height):
                    boundary = True

        if boundary:
            flush_current()
            current = [pm]
            if pm.page_of_total:
                current_total = pm.page_of_total[1]
                next_expected_idx = pm.page_of_total[0] + 1
            else:
                current_total = None
                next_expected_idx = None
        else:
            current.append(pm)
            if pm.page_of_total and next_expected_idx is not None:
                next_expected_idx = pm.page_of_total[0] + 1

    flush_current()

    # Output paths
    out_dir_final = out_dir or os.path.dirname(results_json_path)
    os.makedirs(out_dir_final, exist_ok=True)
    jsonl_path = os.path.join(out_dir_final, "structured_documents.jsonl")
    index_path = os.path.join(out_dir_final, "documents_index.json")

    # Sort documents chronologically (ascending) by document_date (chosen) then first page
    def _sort_key(d: DocumentRecord) -> Tuple[str, int]:
        sd = d.document_date or d.sort_date or "9999-99-99"
        first_page = min(d.pages) if d.pages else 999999
        return (str(sd), int(first_page))

    # Assign chronological IDs by oldest-first
    docs_by_date_asc = sorted(documents, key=_sort_key)
    for idx, d in enumerate(docs_by_date_asc, start=1):
        d.chron_id = int(idx)
    # Output newest-first while keeping chron_id from oldest-first
    documents_sorted = sorted(documents, key=_sort_key, reverse=True)

    # Write JSONL and index
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in documents_sorted:
            f.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")

    index: Dict[str, Dict[str, Any]] = {}
    for d in documents:
        index[d.document_id] = {
            "original_id": d.original_id,
            "chron_id": d.chron_id,
            "document_type": d.document_type,
            "document_date": d.document_date,
            "issuer": d.issuer,
            "pages": d.pages,
            "source": d.source,
            "patient_name": d.patient_name,
        }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return jsonl_path, index_path


__all__ = [
    "build_ocr_documents",
]


