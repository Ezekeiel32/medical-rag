import json
import os
from typing import Dict, List, Any, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_st_model() -> SentenceTransformer:
    preferred = _device()
    try:
        return SentenceTransformer(EMBED_MODEL, device=preferred)
    except Exception:
        # Fallback to CPU if CUDA not usable at runtime
        return SentenceTransformer(EMBED_MODEL, device="cpu")


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _metadata_path(base_dir: str) -> str:
    return os.path.join(base_dir, "meta.jsonl")


def _index_path(base_dir: str) -> str:
    return os.path.join(base_dir, "index.faiss")


def _model_path(base_dir: str) -> str:
    return os.path.join(base_dir, "model.json")


def _vectors_path(base_dir: str) -> str:
    return os.path.join(base_dir, "vectors.npy")


def build_medication_corpus(index_json_path: str) -> List[Dict[str, Any]]:
    data = json.load(open(index_json_path))
    rows: List[Dict[str, Any]] = []
    for patient_id, bucket in data.items():
        for med in (bucket.get("MedicationRequest") or []):
            med_name = None
            codeable = med.get("medicationCodeableConcept")
            if isinstance(codeable, dict):
                med_name = codeable.get("text")
                coding = codeable.get("coding") or []
                if not med_name and coding and isinstance(coding, list):
                    med_name = coding[0].get("display") or coding[0].get("code")
            dosage_parts: List[str] = []
            for di in med.get("dosageInstruction", []) or []:
                if isinstance(di, dict) and di.get("text"):
                    dosage_parts.append(di["text"])
            text = " | ".join(
                part for part in [
                    f"Medication: {med_name}" if med_name else None,
                    f"Dosage: {'; '.join(dosage_parts)}" if dosage_parts else None,
                    f"Status: {med.get('status')}" if med.get("status") else None,
                    f"AuthoredOn: {med.get('authoredOn')}" if med.get("authoredOn") else None,
                ]
                if part
            )
            rows.append({
                "patient_id": patient_id,
                "kind": "medication",
                "text": text,
            })
    return rows


def build_embeddings(data_dir: str, embed_dir_name: str = "embeddings") -> Tuple[str, str]:
    index_json_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_json_path):
        raise FileNotFoundError(f"Index JSON not found at {index_json_path}. Run indexing first.")

    embed_dir = os.path.join(data_dir, embed_dir_name)
    _ensure_dir(embed_dir)

    rows = build_medication_corpus(index_json_path)
    texts = [r["text"] for r in rows]
    model = _load_st_model()
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]

    # Build FAISS index (cosine via IP + normalized vectors)
    index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexIDMap2(index)
    ids = np.arange(len(embeddings), dtype=np.int64)
    id_index.add_with_ids(embeddings.astype(np.float32), ids)
    faiss.write_index(id_index, _index_path(embed_dir))

    with open(_metadata_path(embed_dir), "w") as f:
        for i, row in enumerate(rows):
            row_out = dict(row)
            row_out["row_id"] = int(i)
            f.write(json.dumps(row_out) + "\n")

    with open(_model_path(embed_dir), "w") as f:
        json.dump({"model": EMBED_MODEL, "dim": int(dim)}, f)

    # Persist normalized embeddings for patient-filtered retrieval
    np.save(_vectors_path(embed_dir), embeddings.astype(np.float32))

    return _index_path(embed_dir), _metadata_path(embed_dir)


def _load_index_and_meta(embed_dir: str) -> Tuple[faiss.IndexIDMap2, List[Dict[str, Any]]]:
    index = faiss.read_index(_index_path(embed_dir))
    meta: List[Dict[str, Any]] = []
    with open(_metadata_path(embed_dir), "r") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return index, meta


def _patient_filter(meta: List[Dict[str, Any]], ids: np.ndarray, patient_id: str) -> List[Dict[str, Any]]:
    id_set = set(int(x) for x in ids.tolist())
    results: List[Dict[str, Any]] = []
    for m in meta:
        if int(m.get("row_id", -1)) in id_set and m.get("patient_id") == patient_id:
            results.append(m)
    return results


def rag_query_medications(data_dir: str, patient_id: str, query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
    embed_dir = os.path.join(data_dir, "embeddings")
    index, meta = _load_index_and_meta(embed_dir)

    # Prefer in-memory filtered similarity if vectors are available
    vectors_path = _vectors_path(embed_dir)
    model = _load_st_model()
    query = f"patient:{patient_id} medications: {query_text}"
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    if os.path.exists(vectors_path):
        vectors = np.load(vectors_path)
        # Build mask for this patient
        patient_row_ids: List[int] = [int(m["row_id"]) for m in meta if m.get("patient_id") == patient_id]
        if not patient_row_ids:
            return []
        patient_mat = vectors[np.array(patient_row_ids, dtype=np.int64)]  # normalized
        sims = (patient_mat @ q_emb[0]).astype(np.float32)
        top_idx = np.argsort(-sims)[:top_k]
        ranked_rows = [patient_row_ids[i] for i in top_idx]
        row_to_meta = {int(m["row_id"]): m for m in meta}
        return [row_to_meta[r] for r in ranked_rows]

    # Fallback to FAISS search then filter by patient
    scores, ids = index.search(q_emb, top_k * 10)
    flat_ids = ids[0]
    filtered = _patient_filter(meta, flat_ids, patient_id)
    return filtered[:top_k]


