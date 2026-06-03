from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from shared.api_types import ImportJob, PaperRecord

from ..config import JOBS_DIR, METADATA_FILE, PAPERS_DIR, TEXTS_DIR, ensure_data_dirs
from .utils import model_dump, read_json, write_json


class StorageManager:
    def __init__(self) -> None:
        ensure_data_dirs()

    def list_papers(self, keyword: Optional[str] = None, year: Optional[int] = None, topic: Optional[str] = None) -> List[PaperRecord]:
        papers = self.load_all_papers()
        if keyword:
            keyword_lower = keyword.lower()
            papers = [
                paper for paper in papers
                if keyword_lower in paper.title.lower()
                or any(keyword_lower in kw.lower() for kw in paper.keywords)
                or keyword_lower in paper.abstract.lower()
            ]
        if year:
            papers = [paper for paper in papers if paper.year == year]
        if topic:
            papers = [paper for paper in papers if (paper.topic or "").lower() == topic.lower()]
        papers.sort(key=lambda paper: paper.created_at, reverse=True)
        return papers

    def load_all_papers(self) -> List[PaperRecord]:
        if not METADATA_FILE.exists():
            return []
        papers: List[PaperRecord] = []
        with METADATA_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                papers.append(PaperRecord(**json.loads(line)))
        return papers

    def list_jobs(self) -> List[ImportJob]:
        jobs: List[ImportJob] = []
        for path in sorted(JOBS_DIR.glob("*.json")):
            try:
                jobs.append(ImportJob(**read_json(path, default={})))
            except Exception:
                continue
        return jobs

    def save_papers(self, papers: List[PaperRecord]) -> None:
        ensure_data_dirs()
        with METADATA_FILE.open("w", encoding="utf-8") as fh:
            for paper in papers:
                fh.write(json.dumps(model_dump(paper), ensure_ascii=False) + "\n")

    def upsert_paper(self, paper: PaperRecord) -> None:
        papers = self.load_all_papers()
        replaced = False
        for index, existing in enumerate(papers):
            if existing.paper_id == paper.paper_id:
                papers[index] = paper
                replaced = True
                break
        if not replaced:
            papers.append(paper)
        self.save_papers(papers)

    def get_paper(self, paper_id: str) -> Optional[PaperRecord]:
        for paper in self.load_all_papers():
            if paper.paper_id == paper_id:
                return paper
        return None

    def save_text_payload(self, paper_id: str, payload: Dict) -> None:
        path = TEXTS_DIR / f"{paper_id}.json"
        write_json(path, payload)

    def load_text_payload(self, paper_id: str) -> Dict:
        path = TEXTS_DIR / f"{paper_id}.json"
        return read_json(path, default={})

    def save_job(self, job: ImportJob) -> None:
        path = JOBS_DIR / f"{job.job_id}.json"
        write_json(path, model_dump(job))

    def load_job(self, job_id: str) -> Optional[ImportJob]:
        path = JOBS_DIR / f"{job_id}.json"
        if not path.exists():
            return None
        return ImportJob(**read_json(path, default={}))

    def save_pdf_copy(self, paper_id: str, source_path: Path) -> Path:
        target = PAPERS_DIR / f"{paper_id}{source_path.suffix.lower()}"
        target.write_bytes(source_path.read_bytes())
        return target

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
