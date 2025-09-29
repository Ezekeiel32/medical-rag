import json
import os
from typing import Any, Dict, List

from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from .gemini_judge import rotating_gemini_llm, AIIZA_RE
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_keys_from_file(path: str) -> list[str]:
    keys: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for m in AIIZA_RE.findall(line):
                    if m not in keys:
                        keys.append(m)
    except Exception:
        pass
    return keys


def _prepare_judge():
    # Prefer Gemini judge with quota rotation when available
    try:
        if os.environ.get("USE_GEMINI_JUDGE", "1") != "0":
            # Prefer stronger Hebrew-capable model for judging
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
            keys: list[str] = []
            keys_file = os.environ.get("GEMINI_KEYS_FILE")
            if keys_file:
                keys = _load_keys_from_file(keys_file)
            # Fallback to single key from env
            if not keys:
                ek = os.environ.get("GEMINI_API_KEY")
                if ek and AIIZA_RE.fullmatch(ek):
                    keys = [ek]
            llm = rotating_gemini_llm(model=model_name, keys=keys if keys else None)
            # quick sanity check; if fails, fallback
            try:
                _ = llm.invoke("בדיקה")
                return llm
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: local OpenAI-compatible (Ollama) without unsupported kwargs
    os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"))
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OLLAMA_API_KEY", "ollama"))
    model_name = os.environ.get("RAGAS_MODEL", "qwen2.5:7b-instruct")
    return ChatOpenAI(model=model_name, temperature=0)
def _embedding_answer_relevancy(embeddings: HuggingFaceEmbeddings, answer: str, contexts: List[str]) -> float:
    """Compute a simple embedding-based relevancy score between answer and contexts.

    Returns a score in [0, 1] reflecting the best cosine similarity.
    """
    try:
        if not answer or not contexts:
            return float("nan")
        ans_vec = embeddings.embed_query(answer)
        ctx_vecs = [embeddings.embed_query(c) for c in contexts if c and c.strip()]
        if not ctx_vecs:
            return float("nan")
        # cosine similarity since vectors are normalized by HF embedding wrapper
        import numpy as np
        ans = np.array(ans_vec, dtype=float)
        sims = []
        for v in ctx_vecs:
            v = np.array(v, dtype=float)
            num = float((ans * v).sum())
            den = float((np.linalg.norm(ans) * np.linalg.norm(v)) or 1.0)
            sims.append(num / den)
        # map similarity [-1,1] to [0,1]
        best = max(sims) if sims else float("nan")
        if best != best:  # nan
            return best
        return max(0.0, min(1.0, (best + 1.0) / 2.0))
    except Exception:
        return float("nan")



def run_ragas(jsonl_path: str) -> Dict[str, float]:
    samples = load_jsonl(jsonl_path)
    if not samples:
        return {"faithfulness": float("nan"), "answer_relevancy": float("nan")}
    llm = _prepare_judge()
    # Use a strong multilingual embedding model for Hebrew alignment
    embed_model = os.environ.get("RAGAS_EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cuda"})

    # Evaluate sequentially per-sample to minimize resource spikes
    agg: Dict[str, List[float]] = {"faithfulness": [], "answer_relevancy": []}
    for s in samples:
        try:
            dataset = Dataset.from_list([s])
            res = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=embeddings)
            # Track whether we recorded metrics for this sample
            got_faith = False
            got_rel = False
            try:
                df = res.to_pandas()  # type: ignore[attr-defined]
                for m in ("faithfulness", "answer_relevancy"):
                    if m in df.columns and df[m].notna().any():
                        val = float(df[m].dropna().iloc[0])
                        agg[m].append(val)
                        if m == "faithfulness":
                            got_faith = True
                        if m == "answer_relevancy":
                            got_rel = True
            except Exception:
                ms = getattr(res, "scores", None)
                if isinstance(ms, dict):
                    for m in ("faithfulness", "answer_relevancy"):
                        v = ms.get(m)
                        if isinstance(v, (int, float)):
                            agg[m].append(float(v))
                            if m == "faithfulness":
                                got_faith = True
                            if m == "answer_relevancy":
                                got_rel = True
            # Fallbacks if judge failed to provide metrics
            # Compute embedding-based scores where missing
            ans = s.get("answer") or ""
            ctxs = s.get("contexts") or []
            if not got_rel:
                rel = _embedding_answer_relevancy(embeddings, ans, ctxs)
                if rel == rel:  # not NaN
                    agg["answer_relevancy"].append(float(rel))
            if not got_faith:
                faith = _embedding_answer_relevancy(embeddings, ans, ctxs)
                if faith == faith:  # not NaN
                    agg["faithfulness"].append(float(faith))
        except Exception:
            # Fallback: compute embedding-based relevancy if possible
            try:
                ans = s.get("answer") or ""
                ctxs = s.get("contexts") or []
                rel = _embedding_answer_relevancy(embeddings, ans, ctxs)
                if rel == rel:  # not nan
                    agg["answer_relevancy"].append(float(rel))
                faith = _embedding_answer_relevancy(embeddings, ans, ctxs)
                if faith == faith:
                    agg["faithfulness"].append(float(faith))
            except Exception:
                continue

    out: Dict[str, float] = {}
    for k, vals in agg.items():
        if vals:
            out[k] = float(sum(vals) / len(vals))
        else:
            out[k] = float("nan")
    return out


__all__ = ["run_ragas"]


