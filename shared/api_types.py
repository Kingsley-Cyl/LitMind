from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    code: int = 0
    message: str = "ok"
    data: Any = None


class ImportRequest(BaseModel):
    directory: str
    enable_online_metadata: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    keywords: List[str] = Field(default_factory=list)
    summary: str = ""
    sections: Dict[str, str] = Field(default_factory=dict)
    pdf_path: str
    language: str = "unknown"
    created_at: str
    topic: Optional[str] = None
    important_sentences: List[str] = Field(default_factory=list)


class ImportJob(BaseModel):
    job_id: str
    status: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_step: str = "pending"
    logs: List[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    paper_id: str
    title: str
    score: float
    abstract_snippet: str
    matched_passage: str
    keywords: List[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    paper_id: str
    title: str
    score: float
    reason: str
