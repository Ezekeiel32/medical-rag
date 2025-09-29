import os
import re
import time
from pathlib import Path
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI


AIIZA_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def _read_text_safe(p: Path, max_bytes: int = 1024 * 1024) -> str:
    try:
        if not p.is_file():
            return ""
        if p.stat().st_size > max_bytes:
            return ""
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def discover_gemini_keys(search_roots: Optional[List[str]] = None, max_files: int = 5000) -> List[str]:
    roots = search_roots or ["/home/chezy/Desktop/cursor/robust_zpe"]
    # allow override
    env_key = os.environ.get("GEMINI_API_KEY")
    keys: List[str] = []
    if env_key and AIIZA_RE.fullmatch(env_key):
        keys.append(env_key)

    scanned = 0
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        # prioritize typical places
        candidates: List[Path] = []
        for sub in [rp / "security", rp / "google", rp]:
            if sub.exists():
                for p in sub.rglob("*"):
                    if p.is_file():
                        candidates.append(p)
        for p in candidates:
            if scanned >= max_files:
                break
            scanned += 1
            txt = _read_text_safe(p)
            if not txt:
                continue
            for m in AIIZA_RE.findall(txt):
                if m not in keys:
                    keys.append(m)
    return keys


def rotating_gemini_llm(model: str = "gemini-2.0-flash-lite", keys: Optional[List[str]] = None) -> ChatGoogleGenerativeAI:
    keys = keys or discover_gemini_keys()
    if not keys:
        # fall back to environment variable even if it doesn't match heuristic
        ek = os.environ.get("GEMINI_API_KEY", "")
        if ek:
            keys = [ek]
    if not keys:
        raise RuntimeError("No Gemini API keys found. Set GEMINI_API_KEY or provide keys in robust_zpe directory.")
    # simple time-based rotation per minute
    idx = int(time.time() // 60) % len(keys)
    api_key = keys[idx]
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


__all__ = ["discover_gemini_keys", "rotating_gemini_llm"]



