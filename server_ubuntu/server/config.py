from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SERVER_ROOT / "data"
PAPERS_DIR = DATA_ROOT / "papers"
TEXTS_DIR = DATA_ROOT / "texts"
JOBS_DIR = DATA_ROOT / "jobs"
UPLOADS_DIR = DATA_ROOT / "uploads"
METADATA_FILE = DATA_ROOT / "metadata.jsonl"
DOC_EMBEDDINGS_FILE = DATA_ROOT / "embeddings.npy"
PASSAGE_EMBEDDINGS_FILE = DATA_ROOT / "passages.npy"
DOC_INDEX_FILE = DATA_ROOT / "faiss.index"
PASSAGE_INDEX_FILE = DATA_ROOT / "passages.faiss.index"
DOC_INDEX_META_FILE = DATA_ROOT / "doc_index_meta.json"
PASSAGE_INDEX_META_FILE = DATA_ROOT / "passage_index_meta.json"
RECOMMEND_DOC_EMBEDDINGS_FILE = DATA_ROOT / "recommend_embeddings.npy"
RECOMMEND_DOC_INDEX_FILE = DATA_ROOT / "recommend.faiss.index"
RECOMMEND_DOC_INDEX_META_FILE = DATA_ROOT / "recommend_index_meta.json"
FINE_TUNING_MODELS_DIR = PROJECT_ROOT / "experiments" / "fine_tuning" / "models"


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, PAPERS_DIR, TEXTS_DIR, JOBS_DIR, UPLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(
    env_name: str,
    default_path: Path | str,
    fallback: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    if isinstance(default_path, Path):
        return str(default_path) if default_path.exists() else fallback
    candidate = Path(default_path)
    return str(candidate) if candidate.exists() else str(default_path or fallback)
