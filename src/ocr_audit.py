import json
import os
import csv
from typing import Any, Dict, List, Tuple


def _reflow_page_text(lines: List[Dict[str, Any]]) -> str:
    # Sort by y then x to ensure reading order
    sorted_lines = sorted(lines or [], key=lambda l: ((l.get("bbox") or [0, 0, 0, 0])[1], (l.get("bbox") or [0, 0, 0, 0])[0]))
    # Estimate typical vertical gap
    ys = [float((ln.get("bbox") or [0, 0, 0, 0])[1]) for ln in sorted_lines]
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)] if len(ys) > 1 else []
    med_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    para_threshold = med_gap * 1.8 if med_gap > 0 else 999999.0

    parts: List[str] = []
    prev_y = None
    for idx, ln in enumerate(sorted_lines):
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        y = float((ln.get("bbox") or [0, 0, 0, 0])[1])
        if prev_y is not None and (y - prev_y) > para_threshold:
            parts.append("\n\n")
        elif parts:
            # Join with space unless previous ended with hyphen-like
            if not parts[-1].endswith(("-", "־", "–")):
                parts.append(" ")
            else:
                # remove the trailing hyphen to stitch words
                parts[-1] = parts[-1][:-1]
        parts.append(t)
        prev_y = y

    # Collapse spaces
    text = "".join(parts)
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def audit_results(ocr_dir: str) -> Dict[str, Any]:
    results_path = os.path.join(ocr_dir, "results.json")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(results_path)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    pages = results.get("pages") or []
    out_pages: List[Dict[str, Any]] = []
    audit_rows: List[List[Any]] = []

    for p in pages:
        page_index = int(p.get("page") or 0)
        width = int(p.get("width") or 0)
        height = int(p.get("height") or 0)
        orig_text = (p.get("text") or "")
        lines = p.get("lines") or []
        aligned_text = _reflow_page_text(lines) if lines else orig_text
        out_pages.append({
            "page": page_index,
            "width": width,
            "height": height,
            "text": aligned_text,
            "lines": lines,
        })
        audit_rows.append([
            page_index,
            len(orig_text or ""),
            len(aligned_text or ""),
            width,
            height,
            (orig_text or "")[:120].replace("\n", " "),
        ])

    aligned = {
        "source": results.get("source"),
        "num_pages": len(out_pages),
        "pages": out_pages,
    }

    out_results_aligned = os.path.join(ocr_dir, "results_aligned.json")
    with open(out_results_aligned, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    # Keep a cleaned twin file identical to aligned for downstream consumers
    out_results_cleaned = os.path.join(ocr_dir, "results_aligned_cleaned.json")
    with open(out_results_cleaned, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    out_full_aligned = os.path.join(ocr_dir, "full_text_aligned.txt")
    with open(out_full_aligned, "w", encoding="utf-8") as f:
        f.write("\n\n".join(p.get("text") or "" for p in out_pages))

    audit_csv = os.path.join(ocr_dir, "audit_pages.csv")
    with open(audit_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "len_orig", "len_aligned", "width", "height", "sample_120"])
        w.writerows(sorted(audit_rows, key=lambda r: r[0]))

    return {
        "results_aligned": out_results_aligned,
        "results_aligned_cleaned": out_results_cleaned,
        "full_text_aligned": out_full_aligned,
        "audit_csv": audit_csv,
        "num_pages": len(out_pages),
    }


__all__ = ["audit_results"]

