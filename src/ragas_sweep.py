import json
import os
from typing import Any, Dict, List, Sequence

from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.embeddings import HuggingFaceEmbeddings

from .case_report import QUESTION_TEMPLATES_HE
from .ocr_ragas_eval import _filters_for_question  # type: ignore
from .rag_ocr import search_ocr_index
from .rag_answer import rag_answer_json
from .ragas_eval import _prepare_judge  # type: ignore


def evaluate_samples(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    llm = _prepare_judge()
    embed_model = os.environ.get("RAGAS_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cuda"})

    agg: Dict[str, List[float]] = {"faithfulness": [], "answer_relevancy": []}
    for s in samples:
        try:
            dataset = Dataset.from_list([s])
            res = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=embeddings)
            try:
                df = res.to_pandas()  # type: ignore[attr-defined]
                for m in ("faithfulness", "answer_relevancy"):
                    if m in df.columns and df[m].notna().any():
                        val = df[m].dropna().iloc[0]
                        if isinstance(val, (int, float)):
                            agg[m].append(float(val))
            except Exception as e:
                print(f"DataFrame error: {e}")
                # Try alternative access
                try:
                    scores = res.scores  # type: ignore
                    if isinstance(scores, dict):
                        for m in ("faithfulness", "answer_relevancy"):
                            v = scores.get(m)
                            if isinstance(v, (int, float)):
                                agg[m].append(float(v))
                except Exception as e2:
                    print(f"Scores access error: {e2}")
                    continue
        except Exception as e:
            print(f"Evaluation error for question: {e}")
            continue

    out: Dict[str, float] = {}
    for k, vals in agg.items():
        out[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def sweep_k(
    ocr_dir: str,
    ks: Sequence[int] = (6, 8, 10, 12, 14),
    model: str = "qwen2.5:7b-instruct",
    base_answer_k: int = 10,
) -> Dict[str, Any]:
    # Generate stable answers once
    answers: Dict[str, str] = {}
    for q in QUESTION_TEMPLATES_HE:
        f = _filters_for_question(q)
        ans = rag_answer_json(
            ocr_dir,
            q,
            top_k=base_answer_k,
            model=model,
            category=f.get("category"),
            doc_type=f.get("document_type"),
        )
        answers[q] = ans.get("answer") or ""

    results: List[Dict[str, Any]] = []
    for k in ks:
        samples: List[Dict[str, Any]] = []
        for q in QUESTION_TEMPLATES_HE:
            f = _filters_for_question(q)
            rows = search_ocr_index(ocr_dir, q, top_k=k, filters={k2: v2 for k2, v2 in f.items() if v2})
            contexts: List[str] = []
            for r in rows:
                txt = (r.get("text") or "").strip()
                if txt and len(txt) > 10:  # Ensure meaningful content
                    contexts.append(txt[:1200])
            # Ensure we have at least some contexts
            if not contexts:
                contexts = ["No relevant context found."]
            samples.append({
                "question": q,
                "answer": answers.get(q, ""),
                "contexts": contexts,
            })
        scores = evaluate_samples(samples)
        scores["k"] = int(k)
        results.append(scores)

    # Pick best k by faithfulness, then answer_relevancy
    def key_fn(d: Dict[str, Any]):
        return (float(d.get("faithfulness") or 0.0), float(d.get("answer_relevancy") or 0.0))

    best = max(results, key=key_fn)
    return {"results": results, "best": best}


__all__ = ["sweep_k"]


