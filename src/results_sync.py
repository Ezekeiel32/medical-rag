import json
import os
from typing import Any, Dict


def _clean_json_control_chars(raw: str) -> str:
    # Mirror backend.clean_json_control_chars behavior
    result_chars = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_string:
            if ch == '\\' and i + 1 < len(raw):
                result_chars.append(ch)
                i += 1
                result_chars.append(raw[i])
                i += 1
                continue
            if ch == '"':
                in_string = False
                result_chars.append(ch)
                i += 1
                continue
            if ch == '\n':
                result_chars.append('\\n')
            elif ch == '\r':
                result_chars.append('\\r')
            elif ch == '\t':
                result_chars.append('\\t')
            elif ord(ch) < 0x20:
                result_chars.append(f'\\u{ord(ch):04x}')
            else:
                result_chars.append(ch)
        else:
            if ch == '"':
                in_string = True
            result_chars.append(ch)
        i += 1
    return ''.join(result_chars)


def _robust_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = _clean_json_control_chars(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: escape stray control chars inside strings (simple pass-through since cleaned already handled most)
        return json.loads(cleaned)


def _read_json(path: str) -> Dict[str, Any]:
    return _robust_load_json(path)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _pages_text(data: Dict[str, Any]) -> str:
    pages = data.get("pages") or []
    return "\n\n".join((p.get("text") or "") for p in pages)


def sync_aligned_files(ocr_dir: str) -> Dict[str, Any]:
    """
    Ensure results_aligned.json and results_aligned_cleaned.json exist and are byte-for-byte identical.
    Preference: if a cleaned file exists, treat it as canonical; otherwise fall back to aligned.

    Also regenerates full_text_aligned.txt from the canonical JSON to guarantee exact match.
    Returns a dict with canonical path and actions performed.
    """
    ra = os.path.join(ocr_dir, "results_aligned.json")
    rc = os.path.join(ocr_dir, "results_aligned_cleaned.json")

    actions = []
    canonical_path = None
    canonical: Dict[str, Any] | None = None

    ra_exists = os.path.isfile(ra)
    rc_exists = os.path.isfile(rc)

    if rc_exists:
        # Try to read cleaned as canonical; if it fails, fall back to aligned
        try:
            canonical = _read_json(rc)
            canonical_path = rc
            actions.append("canonical=cleaned")
        except Exception as e:
            actions.append(f"cleaned_invalid: {type(e).__name__}")
            canonical = None
        if canonical is None and ra_exists:
            try:
                canonical = _read_json(ra)
                canonical_path = ra
                actions.append("canonical=aligned (cleaned invalid)")
            except Exception as e:
                actions.append(f"aligned_invalid: {type(e).__name__}")
        if canonical is None:
            # Attempt to rebuild aligned files from results.json via audit
            try:
                from .ocr_audit import audit_results  # type: ignore
                audit_results(ocr_dir)
                actions.append("rebuilt aligned from results.json via audit")
                # Try reading freshly written aligned
                canonical = _read_json(os.path.join(ocr_dir, "results_aligned.json"))
                canonical_path = os.path.join(ocr_dir, "results_aligned.json")
                actions.append("canonical=aligned (after rebuild)")
            except Exception as e:
                raise FileNotFoundError(f"No valid canonical results JSON found and rebuild failed: {e}")
        # Ensure aligned matches canonical
        try:
            aligned = _read_json(ra) if ra_exists else None
        except Exception:
            aligned = None
        if aligned != canonical:
            _write_json(ra, canonical)
            actions.append(("updated " if ra_exists else "created ") + "results_aligned.json from canonical")
        # Ensure cleaned matches canonical (overwrite even if existed but malformed)
        _write_json(rc, canonical)
        actions.append("updated results_aligned_cleaned.json from canonical")
    elif ra_exists:
        canonical_path = ra
        try:
            canonical = _read_json(ra)
            actions.append("canonical=aligned")
        except Exception:
            # Attempt rebuild from results.json
            try:
                from .ocr_audit import audit_results  # type: ignore
                audit_results(ocr_dir)
                actions.append("rebuilt aligned from results.json via audit")
                canonical = _read_json(ra)
            except Exception as e:
                raise FileNotFoundError(f"Aligned invalid and rebuild failed: {e}")
        _write_json(rc, canonical)
        actions.append("created results_aligned_cleaned.json from aligned")
    else:
        raise FileNotFoundError("Neither results_aligned_cleaned.json nor results_aligned.json found in ocr_dir")

    # Regenerate full_text_aligned.txt and full_text.txt from canonical
    full_text_path = os.path.join(ocr_dir, "full_text_aligned.txt")
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(_pages_text(canonical))
    actions.append("wrote full_text_aligned.txt from canonical")

    full_text_plain = os.path.join(ocr_dir, "full_text.txt")
    with open(full_text_plain, "w", encoding="utf-8") as f:
        f.write(_pages_text(canonical))
    actions.append("wrote full_text.txt from canonical")

    return {
        "canonical": canonical_path,
        "actions": actions,
        "full_text": full_text_path,
        "full_text_plain": full_text_plain,
    }


__all__ = ["sync_aligned_files"]


