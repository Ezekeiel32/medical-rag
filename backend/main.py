#!/usr/bin/env python3
"""
FastAPI backend for MediRAG system.
Provides REST API endpoints for the React frontend.
"""

import os
import sys
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

"""RAG integration
We integrate with the Hebrew medical OCR RAG system built in src/.
"""
# Add project root to path for importing RAG modules
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

try:
    from src.rag_answer import rag_answer_json  # type: ignore
    from src.case_report import build_case_report  # type: ignore
    from src.extract_patient_name import extract_patient_info  # type: ignore
    RAG_AVAILABLE = True
    print(" RAG modules loaded successfully")
    
    # Initialize Gemini quota system on startup
    try:
        api_keys_file = str(Path(__file__).parent.parent / "api_gemini15.txt")
        if os.path.exists(api_keys_file):
            # Import and setup quota keys
            import importlib.util
            setup_quota_spec = importlib.util.spec_from_file_location("setup_quota_keys", 
                str(Path(__file__).parent.parent / "setup_quota_keys.py"))
            setup_quota_module = importlib.util.module_from_spec(setup_quota_spec)
            setup_quota_spec.loader.exec_module(setup_quota_module)
            
            # Load and setup API keys
            keys = setup_quota_module.load_api_keys_from_file(api_keys_file)
            if keys and len(keys) > 0:
                success = setup_quota_module.setup_quota_environment(keys)
                if success:
                    print(f" Gemini quota system initialized with {len(keys)} API keys")
                else:
                    print(" WARNING: Failed to setup Gemini quota environment")
            else:
                print(" WARNING: No Gemini API keys found in api_gemini15.txt")
        else:
            print(f" WARNING: Gemini API keys file not found: {api_keys_file}")
    except Exception as gemini_setup_error:
        print(f" WARNING: Gemini quota setup failed: {gemini_setup_error}")
        # Try setting a single key if available from environment
        if not os.getenv('GEMINI_API_KEY') and os.path.exists(api_keys_file):
            try:
                with open(api_keys_file, 'r') as f:
                    first_key = f.readline().strip()
                    if first_key and first_key.startswith('AIza'):
                        os.environ['GEMINI_API_KEY'] = first_key
                        print(" Set single Gemini API key as fallback")
            except Exception:
                pass
    
except Exception as import_error:  # pragma: no cover
    rag_answer_json = None  # type: ignore
    build_case_report = None  # type: ignore
    RAG_AVAILABLE = False
    print(f"  RAG modules not available: {import_error}")

app = FastAPI(
    title="MediRAG API",
    description="Medical RAG system API for patient document analysis",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "https://contentcraft-ai-l5m5m.web.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Patients registry file (simple JSON-based DB)
PATIENTS_DB_PATH = str(Path(__file__).parent.parent / "patients_db.json")

def _load_patients_db() -> Dict[str, Dict[str, Any]]:
    try:
        if os.path.exists(PATIENTS_DB_PATH):
            with open(PATIENTS_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load patients DB: {e}")
    return {}

def _save_patients_db(db: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(PATIENTS_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save patients DB: {e}")

# Global RAG system instance (service)
rag_system = None

# Simple in-memory ingest status tracking
INGEST_STATUS: Dict[str, str] = {}

# Simple in-memory OCR job tracking (for direct PDF → results API)
OCR_JOBS: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class PatientSummary(BaseModel):
    idNumber: str
    name: str
    docCount: int
    lastUpdatedIso: str

class AnswerSource(BaseModel):
    document_id: str
    document_type: str
    document_date: Optional[str]
    pages: List[int]
    quote: str

class PageArtifact(BaseModel):
    page_number: int
    content: str
    metadata: Dict[str, Any]

class PDFArtifact(BaseModel):
    name: str
    path: str
    type: str

class ResponseArtifacts(BaseModel):
    pdfs: List[PDFArtifact]
    pages: List[PageArtifact]

class AnswerJSON(BaseModel):
    question: str
    answer: str
    sources: List[AnswerSource]
    artifacts: Optional[ResponseArtifacts] = None

class ChronologyItem(BaseModel):
    document_name: str
    document_date: Optional[str]
    summary: str
    quote: str
    analysis: str
    document_id: str
    pages: List[int]

class CaseReportJSON(BaseModel):
    answers: List[AnswerJSON]
    chronology: List[ChronologyItem]

class RagAnswerRequest(BaseModel):
    patientId: str
    question: str

class CaseReportRequest(BaseModel):
    patientId: str

class SignedUrlResponse(BaseModel):
    url: str

CANONICAL_QUESTIONS = [
    "מהו מצבו התפקודי העדכני של המבוטח?",
    "מהי המלצת הרופא התעסוקתית לגבי חזרה לעבודה?",
    "האם קיים אישור מחלה בתוקף?"
]

class RAGService:
    """Hebrew Medical OCR RAG Service - integrates with the developed RAG system."""

    def __init__(self, ocr_dir: str) -> None:
        # Base root for per-patient OCR directories. Default patient-less dir is also supported.
        self.base_root = ocr_dir
        self.ocr_dir = ocr_dir
        self.embed_dir = os.path.join(self.ocr_dir, "ocr_embeddings")

    def _patient_dir(self, patient_id: Optional[str]) -> str:
        if not patient_id:
            return self.ocr_dir
        pdir = Path(self.base_root) / patient_id
        return str(pdir) if pdir.exists() else self.ocr_dir

    def _ensure_embeddings(self) -> None:
        """Ensure OCR embeddings exist."""
        if not RAG_AVAILABLE:
            return
        index_path = os.path.join(self.embed_dir, "index.faiss")
        if not os.path.exists(index_path):
            # Build OCR index if missing
            from src.rag_ocr import index_ocr_corpus  # type: ignore
            try:
                index_ocr_corpus(self.ocr_dir)
            except Exception as e:
                print(f"Failed to build OCR index: {e}")

    def patients_from_index(self) -> List[PatientSummary]:
        """Get patient summaries from registry; fallback to single-dir CSV if registry missing."""
        db = _load_patients_db()
        out: List[PatientSummary] = []
        if db:
            for pid, meta in db.items():
                name = str(meta.get("name") or "מבוטח")
                try:
                    doc_count = int(meta.get("docCount") or 0)
                except Exception:
                    doc_count = 0
                last = str(meta.get("lastUpdatedIso") or (datetime.now().isoformat() + "Z"))
                out.append(PatientSummary(idNumber=pid, name=name, docCount=doc_count, lastUpdatedIso=last))
        # Always also include single-directory fallback as legacy patient if present
        csv_path = os.path.join(self.ocr_dir, "clean_documents.csv")
        try:
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    first_row = df.iloc[0]
                    patient_name = str(first_row.get("patient_name", "חביבאללה סאמי"))
                    doc_count = len(df)
                    latest_date = df["document_date"].max() if "document_date" in df.columns else None
                    # Legacy id preserved for compatibility
                    legacy_id = "patient_1"
                    if not any(p.idNumber == legacy_id for p in out):
                        out.append(PatientSummary(
                            idNumber=legacy_id,
                            name=patient_name,
                            docCount=doc_count,
                            lastUpdatedIso=f"{latest_date}T00:00:00Z" if latest_date else datetime.now().isoformat() + "Z"
                        ))
        except Exception as e:
            print(f"Error reading legacy patient data: {e}")
        return out

    def answer_hebrew_medical_query(self, patient_id: str, question: str) -> Dict[str, Any]:
        """Answer Hebrew medical query using the developed OCR RAG system."""
        if not RAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        # Switch to patient's directory if exists
        self.ocr_dir = self._patient_dir(patient_id)
        self.embed_dir = os.path.join(self.ocr_dir, "ocr_embeddings")
        self._ensure_embeddings()
        # If patient index missing, fallback to legacy default corpus instead of error
        index_path = os.path.join(self.embed_dir, "index.faiss")
        if not os.path.exists(index_path):
            legacy_dir = self.base_root  # default ocr_out
            legacy_index = os.path.join(legacy_dir, "ocr_embeddings", "index.faiss")
            if os.path.exists(legacy_index):
                self.ocr_dir = legacy_dir
                self.embed_dir = os.path.join(self.ocr_dir, "ocr_embeddings")
            else:
                raise HTTPException(status_code=400, detail="No documents indexed for this patient yet. Please upload PDFs first.")
        try:
            # Use the real RAG system
            result = rag_answer_json(self.ocr_dir, question)
            return result
        except Exception as e:
            print(f"RAG query error: {e}")
            raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

    def generate_medical_case_report(self, patient_id: str) -> Dict[str, Any]:
        """Generate full case report using the developed system."""
        if not RAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        try:
            # Switch to patient's directory if exists
            self.ocr_dir = self._patient_dir(patient_id)
            self.embed_dir = os.path.join(self.ocr_dir, "ocr_embeddings")
            # Ensure index or at least allow report generation without vector index
            # Use the real case report generator
            report = build_case_report(self.ocr_dir)
            return report
        except Exception as e:
            print(f"Case report error: {e}")
            raise HTTPException(status_code=500, detail=f"Case report failed: {str(e)}")


# Global instance
rag_service: Optional[RAGService] = None

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "MediRAG API is running", "version": "1.0.0"}

@app.get("/api/patients", response_model=Dict[str, List[PatientSummary]])
async def get_patients(search: str = ""):
    """Get list of patients from OCR data."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)

    patients = rag_service.patients_from_index()
    if search:
        s = search.lower()
        patients = [p for p in patients if s in p.name.lower() or s in p.idNumber]
    return {"patients": patients}


# Create patient and upload documents endpoints
class CreatePatientRequest(BaseModel):
    idNumber: Optional[str] = None
    name: Optional[str] = None

@app.post("/api/patients", response_model=PatientSummary)
async def create_patient(req: CreatePatientRequest):
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    db = _load_patients_db()
    pid = (req.idNumber or ("patient_" + datetime.now().strftime("%Y%m%d%H%M%S"))).strip()
    if pid in db:
        raise HTTPException(status_code=400, detail="Patient already exists")
    # Create directory for patient OCR data
    pdir = Path(rag_service.base_root) / pid
    pdir.mkdir(parents=True, exist_ok=True)
    db[pid] = {
        "name": (req.name or "מבוטח חדש").strip(),
        "docCount": 0,
        "lastUpdatedIso": datetime.now().isoformat() + "Z",
    }
    _save_patients_db(db)
    return PatientSummary(idNumber=pid, name=db[pid]["name"], docCount=0, lastUpdatedIso=db[pid]["lastUpdatedIso"])


@app.post("/api/patients/{patient_id}/documents")
async def upload_documents(
    patient_id: str,
    files: List[UploadFile] = File(...),
    bidi: str = Form("visual"),
    background_tasks: BackgroundTasks = None,
):
    """Upload one or more PDFs for a patient, then automatically run OCR+index in background."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    db = _load_patients_db()
    if patient_id not in db:
        raise HTTPException(status_code=404, detail="Patient not found")
    pdir = Path(rag_service.base_root) / patient_id
    pdir.mkdir(parents=True, exist_ok=True)
    pdf_dir = pdir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    processed: List[str] = []
    for uf in files:
        if not uf.filename or not uf.filename.lower().endswith(".pdf"):
            continue
        out_pdf = pdf_dir / uf.filename
        with open(out_pdf, 'wb') as out:
            out.write(await uf.read())
        processed.append(uf.filename)

    # Kick off background ingest (OCR + structuring + indexing)
    try:
        INGEST_STATUS[patient_id] = "queued"
        if background_tasks is not None:
            background_tasks.add_task(run_patient_ingest_sync, patient_id, bidi)
        else:
            # Fallback if no BackgroundTasks injected
            asyncio.create_task(run_patient_ingest_async(patient_id, bidi))
        started = True
    except Exception as e:
        print(f"Failed to enqueue ingest: {e}")
        INGEST_STATUS[patient_id] = f"error: {e}"
        started = False

    return {
        "patientId": patient_id,
        "processed": processed,
        "ingestStarted": bool(started),
        "status": INGEST_STATUS.get(patient_id, "unknown"),
        "note": "OCR ingest is running in background; the index will update automatically when done.",
    }

# -----------------------------
# Direct OCR API (PDF → JSON)
# -----------------------------
def _ensure_uploads_root() -> Path:
    root = Path(__file__).parent.parent / "ocr_out" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root

def _make_ocr_job(out_dir: Path) -> str:
    job_id = str(uuid.uuid4())
    OCR_JOBS[job_id] = {
        "status": "queued",
        "out_dir": str(out_dir),
        "results_json": None,
        "error": None,
        "started_at": datetime.now().isoformat() + "Z",
        "ended_at": None,
    }
    return job_id

def _run_ocr_job_blocking(job_id: str, pdf_path: Path, out_dir: Path, bidi: str = "visual") -> None:
    """Run OCR on a single uploaded PDF and write results.json with text+bboxes."""
    from src.ocr_pipeline import ocr_pdf_best  # type: ignore
    try:
        OCR_JOBS[job_id]["status"] = "ocr_running"
        out_dir.mkdir(parents=True, exist_ok=True)
        res = ocr_pdf_best(
            pdf_path=str(pdf_path),
            output_dir=str(out_dir),
            dpi=300,
            highres_dpi=0,
            prefer_vector_text=True,
            max_pages=None,
            device="cpu",
            bidi_mode=bidi or "visual",
            preprocess=False,
        )
        results_path = out_dir / "results.json"
        OCR_JOBS[job_id]["results_json"] = str(results_path)
        OCR_JOBS[job_id]["status"] = "done"
        OCR_JOBS[job_id]["ended_at"] = datetime.now().isoformat() + "Z"
    except Exception as e:
        OCR_JOBS[job_id]["status"] = "error"
        OCR_JOBS[job_id]["error"] = str(e)
        OCR_JOBS[job_id]["ended_at"] = datetime.now().isoformat() + "Z"

async def _run_ocr_job_async(job_id: str, pdf_path: Path, out_dir: Path, bidi: str = "visual") -> None:
    await asyncio.to_thread(_run_ocr_job_blocking, job_id, pdf_path, out_dir, bidi)

@app.post("/api/ocr/extract")
async def post_ocr_extract(file: UploadFile = File(...), bidi: str = Form("visual"), background_tasks: BackgroundTasks = None):
    """Upload a single PDF and start OCR extraction (text + bboxes) to results.json. Returns jobId."""
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")
    uploads_root = _ensure_uploads_root()
    ts_dir = uploads_root / datetime.now().strftime("job_%Y%m%d%H%M%S")
    ts_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = ts_dir / file.filename
    with open(pdf_path, "wb") as out:
        out.write(await file.read())
    run_dir = ts_dir / f"run_{pdf_path.stem}"
    job_id = _make_ocr_job(run_dir)
    # Start background OCR
    if background_tasks is not None:
        background_tasks.add_task(_run_ocr_job_blocking, job_id, pdf_path, run_dir, bidi)
    else:
        asyncio.create_task(_run_ocr_job_async(job_id, pdf_path, run_dir, bidi))
    return {"jobId": job_id, "status": OCR_JOBS[job_id]["status"], "outDir": str(run_dir)}

@app.get("/api/ocr/status")
async def get_ocr_status(jobId: str):
    job = OCR_JOBS.get(jobId)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"jobId": jobId, "status": job["status"], "error": job.get("error")}

# -----------------------------
# Integrated Patient Creation with OCR
# -----------------------------
@app.post("/api/patients/create-from-pdf")
async def create_patient_from_pdf(
    file: UploadFile = File(...),
    bidi: str = Form("visual"),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a PDF, run OCR, extract patient name/ID, create patient,
    and set up their isolated vector store.
    """
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")
    
    # Step 1: Create patient directory structure first
    patient_id = f"patient_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    pdir = Path(rag_service.base_root) / patient_id
    pdir.mkdir(parents=True, exist_ok=True)
    pdf_dir = pdir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Save PDF to patient's folder
    final_pdf = pdf_dir / file.filename
    with open(final_pdf, 'wb') as f:
        f.write(await file.read())
    
    try:
        # Step 3: Run full OCR pipeline on the patient's directory
        from src.ocr_pipeline import ocr_pdf_best  # type: ignore
        from src.ocr_audit import audit_results  # type: ignore
        from src.ocr_structuring import build_ocr_documents  # type: ignore
        
        # Run OCR on the PDF in the patient's directory
        ocr_results = ocr_pdf_best(
            pdf_path=str(final_pdf),
            output_dir=str(pdir),
            dpi=300,
            highres_dpi=0,
            prefer_vector_text=True,
            max_pages=None,  # Process entire document
            device="cpu",
            bidi_mode=bidi,
            preprocess=False,
        )
        
        # Step 4: Run audit and alignment on the OCR results
        try:
            audit_results(str(pdir))
            # Build structured documents from aligned results
            aligned_results = os.path.join(pdir, "results_aligned.json")
            if os.path.exists(aligned_results):
                build_ocr_documents(aligned_results, str(pdir))
            else:
                # Fallback to regular results.json
                results_json = os.path.join(pdir, "results.json")
                build_ocr_documents(results_json, str(pdir))
        except Exception as e:
            print(f"Audit/structuring failed: {e}")
            # Continue without audit if it fails
        
        # Step 5: Extract patient name and ID from OCR results
        patient_info = extract_patient_info(ocr_results)
        patient_name = patient_info.get('name')
        patient_id_from_doc = patient_info.get('id')
        
        # If no name found, use filename or default
        if not patient_name:
            patient_name = Path(file.filename).stem.replace('_', ' ').replace('-', ' ')
            # Clean up common patterns
            patient_name = re.sub(r'\d+', '', patient_name).strip()
            if not patient_name:
                patient_name = "מבוטח חדש"
        
        # Step 6: Update patient ID if we found one in the document
        if patient_id_from_doc:
            # Check if patient with this ID already exists
            db = _load_patients_db()
            original_patient_id = patient_id
            patient_id = f"patient_{patient_id_from_doc}"
            
            if patient_id in db:
                # Use timestamp-based ID if conflict
                patient_id = f"{original_patient_id}_{patient_id_from_doc}"
            
            # If we changed the patient ID, move the directory
            if patient_id != original_patient_id:
                new_pdir = Path(rag_service.base_root) / patient_id
                if pdir.exists() and not new_pdir.exists():
                    import shutil
                    shutil.move(str(pdir), str(new_pdir))
                    pdir = new_pdir
        
        # Step 7: Save patient to database
        db = _load_patients_db()
        db[patient_id] = {
            "name": patient_name,
            "docCount": 1,
            "lastUpdatedIso": datetime.now().isoformat() + "Z",
        }
        _save_patients_db(db)
        
        # Step 8: Run indexing in background
        INGEST_STATUS[patient_id] = "queued"
        if background_tasks is not None:
            background_tasks.add_task(run_patient_ingest_sync, patient_id, bidi)
        else:
            asyncio.create_task(run_patient_ingest_async(patient_id, bidi))
        
        return {
            "success": True,
            "patient": {
                "idNumber": patient_id,
                "name": patient_name,
                "docCount": 1,
                "lastUpdatedIso": db[patient_id]["lastUpdatedIso"]
            },
            "extractedInfo": {
                "name": patient_info.get('name'),
                "id": patient_info.get('id'),
                "nameSource": "extracted" if patient_info.get('name') else "filename"
            },
            "ingestStatus": "started",
            "message": f"מבוטח '{patient_name}' נוצר בהצלחה והמסמך בעיבוד"
        }
        
    except Exception as e:
        # Clean up patient directory on error
        if pdir.exists():
            try:
                import shutil
                shutil.rmtree(pdir, ignore_errors=True)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.get("/api/ocr/result")
async def get_ocr_result(jobId: str):
    job = OCR_JOBS.get(jobId)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job.get("results_json"):
        return {"jobId": jobId, "status": job["status"]}
    # Load and return results.json content (includes lines with bboxes)
    try:
        with open(job["results_json"], "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"jobId": jobId, "status": "done", "results": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {e}")

@app.post("/api/rag/answer", response_model=AnswerJSON)
async def post_rag_answer(request: RagAnswerRequest):
    """Answer a specific question about a patient using the Hebrew medical OCR RAG system."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)

    # Use the real RAG system
    result = rag_service.answer_hebrew_medical_query(request.patientId, request.question)
    
    # Convert to API format
    sources: List[AnswerSource] = []
    for source in result.get("sources", []):
        sources.append(AnswerSource(
            document_id=str(source.get("document_id", "")),
            document_type=str(source.get("document_type", "")),
            document_date=source.get("document_date"),
            pages=source.get("pages", []),
            quote=str(source.get("quote", ""))
        ))

    # Convert artifacts
    artifacts = None
    if result.get("artifacts"):
        artifacts_data = result["artifacts"]
        pdfs = [PDFArtifact(**pdf) for pdf in artifacts_data.get("pdfs", [])]
        pages = [PageArtifact(**page) for page in artifacts_data.get("pages", [])]
        artifacts = ResponseArtifacts(pdfs=pdfs, pages=pages)

    # Use aggressive response cleaner
    try:
        from src.response_cleaner import clean_medical_response
        clean_answer = clean_medical_response(result.get("answer", ""))
    except Exception as e:
        print(f"Response cleaning failed: {e}")
        clean_answer = result.get("answer", "")

    return AnswerJSON(
        question=request.question, 
        answer=clean_answer, 
        sources=sources,
        artifacts=artifacts
    )

@app.post("/api/rag/case-report", response_model=CaseReportJSON)
async def post_case_report(request: CaseReportRequest):
    """Generate a full case report using the Hebrew medical RAG system."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)

    # Generate enhanced case report using the improved system
    try:
        from src.enhanced_case_report import build_enhanced_case_report
        report_data = build_enhanced_case_report(rag_service.ocr_dir)
        print("Using enhanced case report generation")
    except Exception as e:
        print(f"Enhanced case report failed, using fallback: {e}")
        report_data = rag_service.generate_medical_case_report(request.patientId)
    
    # Convert answers
    answers: List[AnswerJSON] = []
    for answer_data in report_data.get("answers", []):
        sources = [AnswerSource(
            document_id=str(s.get("document_id", "")),
            document_type=str(s.get("document_type", "")),
            document_date=s.get("document_date"),
            pages=s.get("pages", []),
            quote=str(s.get("quote", ""))
        ) for s in answer_data.get("sources", [])]
        
        # Clean answer for case report too
        try:
            from src.response_cleaner import clean_medical_response
            clean_answer = clean_medical_response(answer_data.get("answer", ""))
        except Exception:
            clean_answer = answer_data.get("answer", "")
            
        answers.append(AnswerJSON(
            question=answer_data.get("question", ""),
            answer=clean_answer,
            sources=sources
        ))

    # Convert chronology
    chronology: List[ChronologyItem] = []
    for chron_item in report_data.get("chronology", []):
        chronology.append(ChronologyItem(
            document_name=str(chron_item.get("document_name", "")),
            document_date=chron_item.get("document_date"),
            summary=str(chron_item.get("summary", "")),
            quote=str(chron_item.get("quote", "")),
            analysis=str(chron_item.get("analysis", "")),
            document_id=str(chron_item.get("document_id", "")),
            pages=chron_item.get("pages", [])
        ))

    return CaseReportJSON(answers=answers, chronology=chronology)

@app.get("/api/documents/pdf")
async def get_pdf_document(request: "Request"):
    """Serve the medical PDF document with proper CORS headers."""
    from fastapi.responses import FileResponse
    
    pdf_path = str(Path(__file__).parent.parent / "med_patient#1.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF document not found")
    
    resp = FileResponse(
        path=pdf_path,
        media_type="application/pdf",
    )
    # Reflect allowed origin to satisfy CORS for localhost/127.0.0.1
    origin = request.headers.get("origin", "")
    allowed = {"http://localhost:5173", "http://127.0.0.1:5173", "https://contentcraft-ai-l5m5m.web.app"}
    if origin in allowed:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

@app.head("/api/documents/pdf")
async def head_pdf_document(request: "Request"):
    """Return headers for the medical PDF document (no body) with CORS."""
    from fastapi.responses import Response
    pdf_path = Path(__file__).parent.parent / "med_patient#1.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF document not found")
    stat = pdf_path.stat()
    headers = {
        "Content-Type": "application/pdf",
        "Content-Length": str(stat.st_size),
        "Accept-Ranges": "bytes",
    }
    origin = request.headers.get("origin", "")
    allowed = {"http://localhost:5173", "http://127.0.0.1:5173", "https://contentcraft-ai-l5m5m.web.app"}
    if origin in allowed:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return Response(status_code=200, headers=headers)

@app.get("/api/documents/signed-url")
async def get_signed_url(document_id: str, request: "Request"):
    """Get signed URL for a specific document.
    Uses the incoming request's base URL to avoid hostname mismatches (e.g., localhost vs 127.0.0.1).
    """
    # Prefer request base URL to avoid IPv6 localhost issues
    from urllib.parse import urljoin
    base = str(request.base_url)  # e.g., http://127.0.0.1:8000/
    url = urljoin(base, "/api/documents/pdf")
    return {"url": url}

@app.get("/api/documents/{document_id}/page/{page}/text")
async def get_document_page_text(document_id: str, page: int):
    """Get full text content for a specific document page."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    
    try:
        # Load results_aligned.json to get page text
        results_path = os.path.join(rag_service.ocr_dir, "results_aligned.json")
        if not os.path.exists(results_path):
            # Fallback to results.json
            results_path = os.path.join(rag_service.ocr_dir, "results.json")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Document results not found")
        
        # Read file content robustly (handles unescaped newlines/tabs in strings)
        try:
            results_data = robust_load_json_file(results_path)
        except json.JSONDecodeError as json_error:
            # Fallback to base results.json if aligned file has malformed strings
            alt_path = os.path.join(rag_service.ocr_dir, "results.json")
            try:
                results_data = robust_load_json_file(alt_path)
            except Exception:
                print(f"JSON decode error for single page: {json_error}")
                raise HTTPException(status_code=500, detail=f"Failed to parse document data: {str(json_error)}")
        
        # Find the specific page
        pages = results_data.get("pages", [])
        if not pages or page < 1 or page > len(pages):
            raise HTTPException(status_code=404, detail="Page not found")
        
        page_data = pages[page - 1]  # 0-indexed
        text_content = page_data.get("text", "")
        
        # If text is empty or too short, try to reconstruct from lines
        if not text_content.strip() or len(text_content.strip()) < 50:
            lines = page_data.get("lines", [])
            if lines:
                line_texts = []
                for line in lines:
                    if isinstance(line, dict):
                        line_text = str(line.get("text", ""))
                        # Clean control characters from line text
                        line_text = clean_json_control_chars(line_text)
                        if line_text.strip():
                            line_texts.append(line_text.strip())
                text_content = "\n".join(line_texts)
        
        # Clean the final text content
        text_content = clean_json_control_chars(text_content)
        
        return {"text": text_content}
        
    except Exception as e:
        print(f"Error getting document page text: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document text: {str(e)}")

def clean_json_control_chars(json_string: str) -> str:
    """Clean invalid control characters from JSON string."""
    import re
    # Remove or replace common problematic control characters
    # Keep only valid JSON control characters: \b, \f, \n, \r, \t
    # Remove other control characters (0x00-0x1F except valid ones)
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_string)
    return cleaned

def escape_unescaped_controls_in_strings(json_string: str) -> str:
    """Escape bare control characters that appear inside JSON strings.

    Some OCR exporters write literal newlines/tabs/CR inside string values
    without escaping them, which breaks JSON parsing. This walks the text and
    escapes those characters only when inside a string context.
    """
    result_chars: List[str] = []
    in_string = False
    is_escaped = False
    for ch in json_string:
        if in_string:
            if is_escaped:
                result_chars.append(ch)
                is_escaped = False
                continue
            if ch == '\\':
                result_chars.append(ch)
                is_escaped = True
                continue
            if ch == '"':
                result_chars.append(ch)
                in_string = False
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
    return ''.join(result_chars)

def robust_load_json_file(path: str) -> Any:
    """Load JSON file tolerantly by cleaning and escaping invalid controls."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    cleaned = clean_json_control_chars(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        escaped = escape_unescaped_controls_in_strings(cleaned)
        return json.loads(escaped)

@app.get("/api/documents/{document_id}/full-text")
async def get_document_full_text(document_id: str):
    """Get full text content for the entire document (all pages)."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    
    try:
        # Load results_aligned.json to get all page text
        results_path = os.path.join(rag_service.ocr_dir, "results_aligned.json")
        if not os.path.exists(results_path):
            # Fallback to results.json
            results_path = os.path.join(rag_service.ocr_dir, "results.json")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Document results not found")
        
        # Read file content robustly (handles unescaped newlines/tabs in strings)
        try:
            results_data = robust_load_json_file(results_path)
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error: {json_error}")
            # Try alternative approach - read full_text_aligned.txt if available
            full_text_path = os.path.join(rag_service.ocr_dir, "full_text_aligned.txt")
            if os.path.exists(full_text_path):
                with open(full_text_path, 'r', encoding='utf-8') as f:
                    full_text_content = f.read()
                return {
                    "text": full_text_content,
                    "total_pages": "unknown",
                    "document_id": document_id,
                    "source": "full_text_aligned.txt"
                }
            else:
                raise HTTPException(status_code=500, detail=f"JSON parsing failed and no fallback available: {str(json_error)}")
        
        # Get all pages
        pages = results_data.get("pages", [])
        if not pages:
            raise HTTPException(status_code=404, detail="No pages found in document")
        
        # Combine all pages into full text
        full_text_parts = []
        for i, page_data in enumerate(pages):
            page_num = i + 1
            text_content = page_data.get("text", "")
            
            # If text is empty or too short, try to reconstruct from lines
            if not text_content.strip() or len(text_content.strip()) < 50:
                lines = page_data.get("lines", [])
                if lines:
                    # Clean text from lines too
                    line_texts = []
                    for line in lines:
                        if isinstance(line, dict):
                            line_text = str(line.get("text", ""))
                            # Clean control characters from line text
                            line_text = clean_json_control_chars(line_text)
                            if line_text.strip():
                                line_texts.append(line_text.strip())
                    text_content = "\n".join(line_texts)
            
            # Clean the text content
            text_content = clean_json_control_chars(text_content)
            
            if text_content.strip():  # Only add non-empty pages
                # Add page separator
                full_text_parts.append(f"\n{'='*60}\nעמוד {page_num}\n{'='*60}\n")
                full_text_parts.append(text_content.strip())
        
        full_text = "\n\n".join(full_text_parts)
        
        return {
            "text": full_text,
            "total_pages": len(pages),
            "document_id": document_id,
            "source": "results_aligned.json"
        }
        
    except Exception as e:
        print(f"Error getting full document text: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get full document text: {str(e)}")

@app.get("/api/documents/results-aligned")
async def get_results_aligned():
    """Get the complete results_aligned.json file."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    
    try:
        results_path = os.path.join(rag_service.ocr_dir, "results_aligned.json")
        if not os.path.exists(results_path):
            # Fallback to results.json
            results_path = os.path.join(rag_service.ocr_dir, "results.json")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Results file not found")
        
        # Use robust loader to handle unescaped controls inside strings
        results_data = robust_load_json_file(results_path)
        
        return results_data
        
    except Exception as e:
        print(f"Error getting results_aligned.json: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get results file: {str(e)}")

# -----------------------------
# Ingest Orchestration (Server)
# -----------------------------
def run_patient_ingest_sync(patient_id: str, bidi: str = "visual") -> None:
    """Synchronous wrapper for background OCR+index ingest of a patient's PDFs."""
    try:
        asyncio.run(run_patient_ingest_async(patient_id, bidi))
    except RuntimeError:
        # If already inside an event loop, run in a fresh loop
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_patient_ingest_async(patient_id, bidi))
        loop.close()

async def run_patient_ingest_async(patient_id: str, bidi: str = "visual") -> None:
    """Async wrapper to execute blocking OCR/IO work in a thread."""
    await asyncio.to_thread(_ingest_patient_blocking, patient_id, bidi)

def _ingest_patient_blocking(patient_id: str, bidi: str = "visual") -> None:
    """Blocking ingestion: run OCR per uploaded PDF, structure outputs, aggregate, and index."""
    global rag_service, INGEST_STATUS
    try:
        INGEST_STATUS[patient_id] = "starting"
        if rag_service is None:
            ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
            rag_service = RAGService(ocr_dir)

        base_root = Path(rag_service.base_root)
        pdir = base_root / patient_id
        pdf_dir = pdir / "pdfs"
        if not pdf_dir.exists():
            INGEST_STATUS[patient_id] = "no_pdfs"
            return

        from src.ocr_pipeline import ocr_pdf_best  # type: ignore
        from src.ocr_audit import audit_results  # type: ignore
        from src.ocr_structuring import build_ocr_documents  # type: ignore
        from src.rag_ocr import index_ocr_corpus  # type: ignore

        # Run OCR for each PDF in patient's folder
        INGEST_STATUS[patient_id] = "ocr_running"
        runs: List[Path] = []
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            run_dir = pdir / f"run_{pdf_file.stem}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                ocr_pdf_best(
                    pdf_path=str(pdf_file),
                    output_dir=str(run_dir),
                    dpi=300,
                    highres_dpi=0,
                    prefer_vector_text=True,
                    max_pages=None,
                    device="cpu",
                    bidi_mode=bidi or "visual",
                    preprocess=False,
                )
            except Exception as e:
                print(f"OCR failed for {pdf_file}: {e}")
                continue

            # Run audit and alignment on the OCR results
            try:
                audit_results(str(run_dir))
            except Exception as e:
                print(f"Audit failed for {pdf_file}: {e}")
                # Continue without audit if it fails

            # Structure single-run results into JSONL
            try:
                results_json = run_dir / "results_aligned.json"
                if not results_json.exists():
                    results_json = run_dir / "results.json"
                build_ocr_documents(str(results_json), str(run_dir))
                runs.append(run_dir)
            except Exception as e:
                print(f"Structuring failed for {pdf_file}: {e}")
                continue

        # Aggregate all per-run JSONLs into patient's root structured_documents.jsonl
        INGEST_STATUS[patient_id] = "aggregating"
        agg_jsonl = pdir / "structured_documents.jsonl"
        with open(agg_jsonl, "w", encoding="utf-8") as agg_out:
            for rdir in runs:
                sj = rdir / "structured_documents.jsonl"
                if sj.exists():
                    with open(sj, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                agg_out.write(line)

        # Build FAISS index for patient root
        INGEST_STATUS[patient_id] = "indexing"
        try:
            index_ocr_corpus(str(pdir))
        except Exception as e:
            print(f"Indexing failed for patient {patient_id}: {e}")
            INGEST_STATUS[patient_id] = f"index_error: {e}"

        # Update registry docCount and timestamp
        db = _load_patients_db()
        try:
            count = 0
            if agg_jsonl.exists():
                with open(agg_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            if patient_id in db:
                db[patient_id]["docCount"] = count
                db[patient_id]["lastUpdatedIso"] = datetime.now().isoformat() + "Z"
                _save_patients_db(db)
        except Exception as e:
            print(f"Failed to update patient doc count after ingest: {e}")

        # Finalize
        if not INGEST_STATUS.get(patient_id, "").startswith("index_error"):
            INGEST_STATUS[patient_id] = "done"

    except Exception as e:
        print(f"Ingest error for patient {patient_id}: {e}")
        INGEST_STATUS[patient_id] = f"error: {e}"

@app.get("/api/patients/{patient_id}/ingest-status")
async def get_ingest_status(patient_id: str) -> Dict[str, str]:
    """Return the current ingest status for a patient."""
    status = INGEST_STATUS.get(patient_id, "unknown")
    return {"patientId": patient_id, "status": status}

@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str):
    """Delete a patient and all associated data."""
    global rag_service
    if rag_service is None:
        ocr_dir = os.environ.get("OCR_DIR", str(Path(__file__).parent.parent / "ocr_out"))
        rag_service = RAGService(ocr_dir)
    
    db = _load_patients_db()
    if patient_id not in db:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    try:
        # Remove patient directory and all contents
        pdir = Path(rag_service.base_root) / patient_id
        if pdir.exists():
            import shutil
            shutil.rmtree(pdir)
        
        # Remove from database
        del db[patient_id]
        _save_patients_db(db)
        
        # Clear any ingest status
        if patient_id in INGEST_STATUS:
            del INGEST_STATUS[patient_id]
        
        return {"message": f"Patient {patient_id} deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete patient: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    
    print(f"Starting MediRAG API server on {host}:{port}")
    print(f"Frontend should be accessible at: http://127.0.0.1:5173")
    print(f"API docs available at: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
