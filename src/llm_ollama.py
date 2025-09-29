import json
import os
from typing import Any, Dict, Optional

import requests


def ollama_generate(
    prompt: str,
    model: str = "gemma2:2b-instruct",
    system: Optional[str] = None,
    temperature: float = 0.2,
    json_only: bool = True,
    endpoint: Optional[str] = None,
) -> Dict[str, Any] | str:
    """
    Call a local Ollama server to generate a completion. If json_only=True, expect a JSON string in 'response'.
    Returns parsed JSON (dict) when json_only else the raw string.
    """
    # Resolve endpoint at call time to respect environment changes
    if not endpoint:
        env_endpoint = os.environ.get("OLLAMA_ENDPOINT")
        if env_endpoint:
            endpoint = env_endpoint
        else:
            host = os.environ.get("OLLAMA_HOST")
            if host:
                if host.startswith("http://") or host.startswith("https://"):
                    base = host
                else:
                    base = f"http://{host}"
                endpoint = f"{base}/api/generate"
            else:
                endpoint = "http://localhost:11434/api/generate"

    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }
    if system:
        body["system"] = system
    if json_only:
        body["format"] = "json"

    resp = requests.post(endpoint, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response") or ""
    if json_only:
        try:
            return json.loads(text)
        except Exception:
            # Fallback: try to find JSON substring
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= start:
                return json.loads(text[start : end + 1])
            raise
    return text


__all__ = [
    "ollama_generate",
]


def ollama_stream(
    prompt: str,
    model: str = "gemma2:2b-instruct",
    system: Optional[str] = None,
    temperature: float = 0.2,
    as_json: bool = False,
    endpoint: Optional[str] = None,
):
    """
    Stream tokens from a local Ollama server. Yields incremental text chunks.
    If as_json=True, requests JSON-formatted output but still yields raw streamed text.
    """
    if not endpoint:
        env_endpoint = os.environ.get("OLLAMA_ENDPOINT")
        if env_endpoint:
            endpoint = env_endpoint
        else:
            host = os.environ.get("OLLAMA_HOST")
            if host:
                if host.startswith("http://") or host.startswith("https://"):
                    base = host
                else:
                    base = f"http://{host}"
                endpoint = f"{base}/api/generate"
            else:
                endpoint = "http://localhost:11434/api/generate"

    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": float(temperature),
        },
    }
    if system:
        body["system"] = system
    if as_json:
        body["format"] = "json"

    with requests.post(endpoint, json=body, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = data.get("response") or ""
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
            except Exception:
                # In case server streams plain text lines
                yield line


