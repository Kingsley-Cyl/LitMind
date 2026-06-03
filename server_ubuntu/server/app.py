from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.api_types import ApiResponse, ImportRequest, SearchRequest

from .config import FINE_TUNING_MODELS_DIR, UPLOADS_DIR, ensure_data_dirs, resolve_model_path
from .services.embedder import Embedder
from .services.import_service import ImportPipelineService
from .services.keyword_extractor import KeywordExtractor
from .services.metadata_enhancer import MetadataEnhancer
from .services.pdf_parser import PDFParser
from .services.retrieval import RetrievalEngine
from .services.storage import StorageManager
from .services.summarizer import Summarizer
from .services.utils import model_dump


ensure_data_dirs()
search_model = resolve_model_path(
    "LITMIND_SEARCH_EMBEDDER",
    FINE_TUNING_MODELS_DIR / "retrieval_scifact" / "best_model",
)
recommend_model = resolve_model_path(
    "LITMIND_RECOMMEND_EMBEDDER",
    FINE_TUNING_MODELS_DIR / "recommendation_scidocs" / "best_model",
)
summary_model = resolve_model_path(
    "LITMIND_SUMMARY_EMBEDDER",
    FINE_TUNING_MODELS_DIR / "summaries_scitldr" / "best_model",
)
keyword_model = resolve_model_path(
    "LITMIND_KEYWORD_EMBEDDER",
    FINE_TUNING_MODELS_DIR / "keywords_inspec" / "best_model",
)
storage = StorageManager()
search_embedder = Embedder(model_name=search_model)
recommend_embedder = Embedder(model_name=recommend_model)
summary_embedder = Embedder(model_name=summary_model)
retrieval = RetrievalEngine(search_embedder, recommend_embedder)
parser = PDFParser()
keyword_extractor = KeywordExtractor(model_name=keyword_model)
metadata_enhancer = MetadataEnhancer()
service = ImportPipelineService(
    storage=storage,
    parser=parser,
    keyword_extractor=keyword_extractor,
    summarizer=Summarizer(),
    embedder=search_embedder,
    retrieval=retrieval,
    metadata_enhancer=metadata_enhancer,
    summary_embedder=summary_embedder,
)
INTERRUPTED_JOBS = service.mark_interrupted_jobs()

app = FastAPI(title="LitMind API", version="0.1.0")


def ok(data=None, message: str = "ok"):
    return ApiResponse(code=0, message=message, data=data)


@app.get("/health")
def health():
    papers = storage.load_all_papers()
    return ok(
        {
            "status": "healthy",
            "paper_count": len(papers),
            "search_embedding_backend": search_embedder.backend_name,
            "recommend_embedding_backend": recommend_embedder.backend_name,
            "summary_embedding_backend": summary_embedder.backend_name,
            "retrieval_backend": retrieval.backend_name,
            "dependency_status": {
                "pymupdf": parser.is_available(),
                "keybert": keyword_extractor.is_keybert_available(),
                "sentence_transformers": search_embedder.is_sentence_transformers_available(),
            },
            "interrupted_jobs_marked": INTERRUPTED_JOBS,
        }
    )


@app.post("/papers/import")
def import_papers(request: ImportRequest):
    try:
        job = service.start_import(request.directory, enable_online_metadata=request.enable_online_metadata)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ok(model_dump(job), message="import started")


@app.post("/papers/import-upload")
async def import_uploaded_papers(
    files: list[UploadFile] = File(...),
    enable_online_metadata: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    upload_dir = UPLOADS_DIR / f"upload_{uuid.uuid4().hex[:12]}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for index, upload in enumerate(files):
        suffix = Path(upload.filename or "").suffix.lower() or ".pdf"
        if suffix != ".pdf":
            continue
        safe_name = Path(upload.filename or f"paper_{index + 1}.pdf").name
        target = upload_dir / f"{index:03d}_{safe_name}"
        content = await upload.read()
        target.write_bytes(content)
        saved_count += 1

    if saved_count == 0:
        raise HTTPException(status_code=400, detail="no pdf files uploaded")

    job = service.start_import(str(upload_dir), enable_online_metadata=enable_online_metadata)
    return ok(model_dump(job), message="upload import started")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return ok(model_dump(job))


@app.get("/papers")
def list_papers(keyword: Optional[str] = None, year: Optional[int] = None, topic: Optional[str] = None):
    papers = [model_dump(paper) for paper in service.list_papers(keyword=keyword, year=year, topic=topic)]
    return ok(papers)


@app.get("/papers/{paper_id}")
def get_paper(paper_id: str):
    paper = service.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    payload = storage.load_text_payload(paper_id)
    data = model_dump(paper)
    data["raw_text"] = payload.get("cleaned_text", "")
    return ok(data)


@app.get("/papers/{paper_id}/pdf")
def get_paper_pdf(paper_id: str):
    paper = service.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    pdf_path = Path(paper.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="pdf file not found")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


@app.post("/search")
def search(request: SearchRequest):
    hits = [model_dump(hit) for hit in service.search(request.query, top_k=request.top_k)]
    return ok(hits)


@app.get("/papers/{paper_id}/recommendations")
def recommendations(paper_id: str, top_k: int = 5):
    paper = service.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    items = [model_dump(item) for item in service.recommend(paper_id, top_k=top_k)]
    return ok(items)


@app.post("/index/rebuild")
def rebuild_index():
    stats = service.rebuild_index()
    return ok(stats, message="index rebuilt")
