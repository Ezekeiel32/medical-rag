# MediRAG – Hebrew Medical OCR RAG System

A production‑grade Retrieval Augmented Generation (RAG) system for Hebrew medical case files with:
- GPU‑accelerated OCR (Surya) + PDF text layer fast‑path
- Page reflow + audit to aligned JSON with per‑page text
- FAISS embeddings over structured per‑document chunks (multilingual‑e5)
- Deterministic extractors for canonical questions (functional status, RTW, sick‑leave validity)
- Evidence‑first answers with citations and full page text accordion
- FastAPI backend + React/Next UI

This README documents setup, usage, and operations end‑to‑end.

---

## 1. Quick Start

### 1.1. Prerequisites
- Python 3.11/3.12
- Node 18+
- CUDA GPU recommended (for Surya OCR & embeddings). CPU works but is slower
- Optional: an Ollama server for local LLMs (`OLLAMA_HOST` or `OLLAMA_ENDPOINT`) if you want model‑generated summaries

### 1.2. Environment
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Environment variables (put in your shell or a `.env` you source before running):
- `OCR_DIR` – where OCR outputs live (default `ocr_out`)
- `OLLAMA_HOST` or `OLLAMA_ENDPOINT` – if using Ollama for model answers (optional)

### 1.3. OCR a PDF (one time per dataset)
```bash
source venv/bin/activate
python -m src.cli_ocr med_patient#1.pdf --out ocr_out --prefer-vector-text --device cuda
```
This will:
- Extract pages (`results.json`)
- Audit + reflow, producing `ocr_out/results_aligned.json` and `ocr_out/audit_pages.csv`
- Build `ocr_out/structured_documents.jsonl` and `ocr_out/documents_index.json`

### 1.4. Build the embedding index
```bash
python - << 'PY'
from src.rag_ocr import index_ocr_corpus
index_ocr_corpus('ocr_out')
PY
```
Creates `ocr_out/ocr_embeddings/{index.faiss, meta.jsonl, vectors.npy}`.

### 1.5. Run the backend API
```bash
source venv/bin/activate
python backend/main.py
```
- API docs: http://127.0.0.1:8000/docs

### 1.6. Run a UI
You have two options in this repository:

Option A – Next.js app (recommended):
```bash
cd MediRAG_UI
npm install
npm run dev
```
Option B – Vite + React TypeScript demo:
```bash
cd MediRAG_UI/frontend
npm install
npm run dev
```
Frontends are configured to talk to the backend via `NEXT_PUBLIC_BACKEND_URL`/`VITE_BACKEND_URL`.

---

## 2. How It Works (High Level)

1) OCR & Text Gathering
- Renders pages with PyMuPDF (300/600 DPI). If the PDF has a selectable text layer, we can prefer it.
- Surya OCR (GPU preferred) recognizes lines; we store `results.json` with text + per‑line boxes.
- A post‑processing audit aligns text to page boundaries and writes `results_aligned.json`.

2) Structuring
- `src/ocr_structuring.py` parses pages → documents. It detects document types, dates, issuers, IDs, and aggregates pages.
- Outputs `structured_documents.jsonl` and an index.

3) Embeddings & Retrieval
- `src/rag_ocr.index_ocr_corpus` chunks document text and multi‑vector features (title/keywords/summary) with multilingual‑e5, builds FAISS.
- `src/rag_ocr.search_ocr_index` does re‑ranking and intent‑aware filtering for canonical questions.

4) Answers & Evidence
- `src/rag_answer.rag_answer_json` builds a prompt context (or uses deterministic extractors) and returns JSON `{question, answer, sources, artifacts}`.
- Deterministic handlers provide rock‑solid logic for:
  - Sick‑leave validity (date ranges)
  - Functional status (initial injury → latest status narrative)
  - Return‑to‑work (occupational doctor priority)
- Every response includes artifacts:
  - PDFs list
  - Relevant pages [{page_number, content, metadata}] – we always return full page text (rebuilt from per‑line OCR if needed).

5) Frontend
- Presents coherent, concise answers in Hebrew, plus citations and a grey accordion with full page text.
- Clicking “הצג” opens the PDF modal on the cited page.

See `docs/ARCHITECTURE.md` for a deep dive.

---

## 3. Canonical Questions
- "מהו מצבו התפקודי העדכני של המבוטח?"
- "מהי המלצת הרופא התעסוקתית לגבי חזרה לעבודה?"
- "האם קיים אישור מחלה בתוקף?"

The system uses deterministic extractors for these, providing reliable, source‑grounded outputs regardless of LLM availability.

---

## 4. Key Modules
- `src/ocr_pipeline.py` – OCR rendering + Surya inference pipeline
- `src/ocr_structuring.py` – per‑page heuristics; groups into logical medical documents
- `src/ocr_audit.py` – boundary audit + page reflow to `results_aligned.json`
- `src/rag_ocr.py` – embeddings + semantic search over structured docs
- `src/rag_answer.py` – answer production, canonical extractors, artifacts generator
- `backend/main.py` – FastAPI API, schema models, endpoints
- `MediRAG_UI/…` – Next.js app (and `MediRAG_UI/frontend` demo app)

---

## 5. Operations

### Refresh embeddings when documents change
```bash
python - << 'PY'
from src.rag_ocr import index_ocr_corpus
index_ocr_corpus('ocr_out')
PY
```

### Re‑OCR a new PDF
```bash
python -m src.cli_ocr NEW.pdf --out ocr_out --prefer-vector-text --device cuda
```

### Environment tips
- GPU recommended. Surya and sentence‑transformers will auto‑detect CUDA.
- For Ollama: set `OLLAMA_HOST=localhost:11434` or `OLLAMA_ENDPOINT=http://localhost:11434/api/generate`.

---

## 6. Tech Stack
- Python: FastAPI, FAISS, sentence‑transformers, Surya OCR, PyMuPDF, Pandas, Pydantic
- JS/TS: Next.js (App Router), Vite + React demo
- Infra: Ollama (optional), CUDA GPU preferred

---

## 7. Troubleshooting
- 500 errors from `/api/rag/answer` → see backend log; common causes: missing OCR files, no embeddings; run OCR & index steps.
- Accordion shows only a few lines → fixed; backend reconstructs full page text from OCR `lines`.
- Missing GPU → system falls back to CPU; expect slower OCR & embedding times.

---

## 8. License & Attribution
Internal project. Surya OCR, sentence‑transformers, FAISS, and other OSS used under their respective licenses.
