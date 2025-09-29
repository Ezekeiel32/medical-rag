# Architecture – MediRAG Hebrew Medical OCR RAG

This document describes the full system architecture, design decisions, and dataflow from PDF to answers with citations.

## Overview

- OCR: PyMuPDF rendering + Surya OCR with per‑line boxes. Vector text (if present) can be preferred.
- Audit & Reflow: aligns OCR outputs per page to `results_aligned.json`; emits `audit_pages.csv` and `full_text_aligned.txt`.
- Structuring: detects document types/dates/ids, groups pages to logical docs, outputs `structured_documents.jsonl` and `documents_index.json`.
- Embeddings: multilingual‑e5 (SentenceTransformers) with FAISS index; multi‑vector per doc (title/keywords/summary/body).
- Retrieval: intent‑aware re‑rank and filters; reconstruct best body chunk for multi‑vector hits.
- Answers: deterministic extractors for canonical questions; JSON output with sources; artifacts always include full page text.
- UI: Web app displays concise answers with citations and a page‑text accordion.

## OCR & Audit

`src/ocr_pipeline.py`
- Renders PDF at 300/600 DPI (downscales overly large high‑res to avoid OOM).
- If `prefer_vector_text`, extracts text layer (bidi visual/logical as chosen). Otherwise uses Surya OCR lines.
- Writes `results.json` with pages[].

`src/ocr_audit.py`
- Reflows text per page, ensures coherent ordering; writes `results_aligned.json`.
- Produces `audit_pages.csv` and `full_text_aligned.txt` for debugging.

## Structuring

`src/ocr_structuring.py`
- Per‑page heuristics: detect document types (מיון/דוח ביקור/סיכום טיפול/תעסוקתי/ועדה/בל250…),
  document dates (label preference rules), issuer, ids, question hints.
- Groups into documents (sequential + header boundary heuristics).
- Assigns `chron_id` by oldest‑first order while writing newest‑first JSONL for convenience.

## Embeddings & Retrieval

`src/rag_ocr.py`
- Chunks body with overlap; also builds auxiliary vectors: title/keywords/summary.
- Stores FAISS `index.faiss`, `meta.jsonl`, `vectors.npy` (normalized) for masked cosine.
- `search_ocr_index` applies:
  - Intent detection (sick‑leave, return‑to‑work, functional status)
  - Boosting and strict filters by document type/phrases
  - Reconstruction from multi‑vector hit to best body chunk
  - Recency and rank stable sort

## Answers & Artifacts

`src/rag_answer.py`
- `rag_answer_json` orchestrates retrieval → answer JSON.
- Deterministic extractors:
  - Sick‑leave validity: extract date ranges and compare to current Jerusalem time (see `src/timezone_utils.py`).
  - Functional status: stitches initial injury evidence (ER work‑accident + fracture) with latest status snippet.
  - Return‑to‑work: favors occupational doctor docs; narrative includes initial injury when available.
- Artifacts (`_create_response_artifacts`):
  - Always attaches `pdfs` and `pages`.
  - Loads `results_aligned.json` if possible; if page `text` is short, reconstructs from `lines` to provide full page text.
  - Falls back to `results_aligned_clean(ed).json` → `results.json` → `structured_documents.jsonl` if needed.

## Backend API

`backend/main.py`
- `/api/rag/answer` → `AnswerJSON {question, answer, sources[], artifacts{pdfs[], pages[]}}`
- `/api/patients` returns a tiny index from `clean_documents.csv`.
- `/api/rag/case-report` builds canonical answers and a chronology from structured docs.

## Frontend

- Next.js app under `MediRAG_UI/` and a Vite demo at `MediRAG_UI/frontend/`.
- The pages accordion shows full page text, while the PDF modal lets users view the original.

## Chronology

`src/case_report.py`
- Sorts structured docs chronologically.
- Provides per‑doc summary, a core quote, and legal relevance analysis (e.g., causality evidence from ER doc, RTW recommendations, physio status).

## GPU & Performance

- OCR and embeddings prefer CUDA; code paths fall back to CPU safely.
- High‑res rendering is bounded to avoid VRAM explosions; single‑page Surya calls reduce peak memory.

## Fail‑safes

- If aligned results contain control chars, code tolerates and falls back gracefully.
- Artifacts always return pages content; if aligned JSON is unavailable, we degrade to structured text.
