from __future__ import annotations

import os
import traceback
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional

from shared.api_types import ImportJob, PaperRecord, RecommendationItem, SearchHit

from .utils import chunk_passages, model_dump


class ImportPipelineService:
    def __init__(self, storage, parser, keyword_extractor, summarizer, embedder, retrieval, metadata_enhancer=None, summary_embedder=None) -> None:
        self.storage = storage
        self.parser = parser
        self.keyword_extractor = keyword_extractor
        self.summarizer = summarizer
        self.embedder = embedder
        self.summary_embedder = summary_embedder or embedder
        self.retrieval = retrieval
        self.metadata_enhancer = metadata_enhancer
        self._lock = Lock()

    def start_import(self, directory: str, enable_online_metadata: bool = False) -> ImportJob:
        source_dir = Path(directory).expanduser().resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {source_dir}")

        job = ImportJob(job_id=uuid.uuid4().hex[:12], status="queued", current_step="queued")
        self.storage.save_job(job)
        thread = Thread(target=self._run_import, args=(job.job_id, source_dir, enable_online_metadata), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> ImportJob | None:
        return self.storage.load_job(job_id)

    def list_papers(self, keyword=None, year=None, topic=None) -> List[PaperRecord]:
        return self.storage.list_papers(keyword=keyword, year=year, topic=topic)

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        return self.storage.get_paper(paper_id)

    def rebuild_index(self) -> Dict[str, object]:
        papers = [model_dump(paper) for paper in self.storage.load_all_papers()]
        payloads = {paper["paper_id"]: self.storage.load_text_payload(paper["paper_id"]) for paper in papers}
        self.retrieval.rebuild(papers, payloads)
        return {
            "paper_count": len(papers),
            "passage_count": len(self.retrieval.passage_meta),
            "backend": self.retrieval.backend_name,
        }

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        document_hits = self.retrieval.search_documents(query, top_k=top_k)
        passage_hits = self.retrieval.search_passages(query, top_k=max(top_k * 3, 10))
        best_passage_by_paper = {}
        for meta, score in passage_hits:
            paper_id = meta["paper_id"]
            if paper_id not in best_passage_by_paper:
                best_passage_by_paper[paper_id] = (meta, score)

        results: List[SearchHit] = []
        for meta, score in document_hits:
            paper = self.storage.get_paper(meta["paper_id"])
            if paper is None:
                continue
            matched_passage = paper.abstract
            passage = best_passage_by_paper.get(paper.paper_id)
            if passage:
                matched_passage = passage[0].get("text", matched_passage)
            results.append(
                SearchHit(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    score=round(score, 4),
                    abstract_snippet=paper.abstract[:240],
                    matched_passage=matched_passage[:500],
                    keywords=paper.keywords,
                )
            )
        return results

    def recommend(self, paper_id: str, top_k: int = 5) -> List[RecommendationItem]:
        hits = self.retrieval.recommend(paper_id, top_k=top_k)
        results = []
        for meta, score in hits:
            paper = self.storage.get_paper(meta["paper_id"])
            if paper is None:
                continue
            reason = paper.keywords[:3]
            results.append(
                RecommendationItem(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    score=round(score, 4),
                    reason=" / ".join(reason) if reason else "High semantic similarity",
                )
            )
        return results

    def _run_import(self, job_id: str, source_dir: Path, enable_online_metadata: bool) -> None:
        pdf_paths = sorted(source_dir.glob("*.pdf"))
        job = self.storage.load_job(job_id)
        if job is None:
            return
        job.status = "running"
        job.total = len(pdf_paths)
        job.current_step = "scanning"
        job.logs.append(f"Found {len(pdf_paths)} PDF files in {source_dir}")
        job.logs.append(
            "Online metadata enhancement is enabled"
            if enable_online_metadata else
            "Online metadata enhancement is disabled"
        )
        self.storage.save_job(job)

        if not pdf_paths:
            job.status = "failed"
            job.current_step = "completed"
            job.logs.append("No PDF files found.")
            self.storage.save_job(job)
            return

        for pdf_path in pdf_paths:
            try:
                self._import_single_pdf(pdf_path, job, enable_online_metadata)
                job.completed += 1
            except Exception as exc:  # pragma: no cover - integration path
                job.failed += 1
                job.status = "running"
                job.logs.append(f"[ERROR] {pdf_path.name}: {exc}")
                job.logs.append(traceback.format_exc(limit=2))
            self.storage.save_job(job)

        try:
            with self._lock:
                job.current_step = "rebuilding_index"
                job.logs.append("Rebuilding vector index")
                self.storage.save_job(job)
                self.rebuild_index()
        except Exception as exc:  # pragma: no cover - integration path
            job.status = "failed"
            job.current_step = "rebuilding_index_failed"
            job.logs.append(f"[ERROR] rebuild index failed: {exc}")
            job.logs.append(traceback.format_exc(limit=2))
            self.storage.save_job(job)
            return

        job.status = "completed" if job.failed == 0 else "completed_with_errors"
        job.current_step = "completed"
        job.logs.append("Import finished.")
        self.storage.save_job(job)

    def _import_single_pdf(self, pdf_path: Path, job: ImportJob, enable_online_metadata: bool) -> None:
        paper_id = uuid.uuid5(uuid.NAMESPACE_URL, str(pdf_path.resolve())).hex[:16]
        job.current_step = f"copying {pdf_path.name}"
        job.logs.append(f"Copying {pdf_path.name} to data/papers")
        self.storage.save_job(job)
        copied_pdf_path = self.storage.save_pdf_copy(paper_id, pdf_path)

        job.current_step = f"parsing {pdf_path.name}"
        job.logs.append(f"Parsing {pdf_path.name}")
        self.storage.save_job(job)
        parsed = self.parser.parse_paper(copied_pdf_path)

        if enable_online_metadata and self.metadata_enhancer is not None:
            job.current_step = f"online_metadata {pdf_path.name}"
            job.logs.append(f"Trying online metadata enhancement for {pdf_path.name}")
            self.storage.save_job(job)
            parsed, source = self.metadata_enhancer.enhance(parsed)
            if source:
                job.logs.append(f"Online metadata matched via {source}")
            else:
                job.logs.append("Online metadata enhancement did not find a better match")

        job.current_step = f"extracting_keywords {pdf_path.name}"
        job.logs.append(f"Extracting keywords for {pdf_path.name}")
        self.storage.save_job(job)
        analysis_text = " ".join([
            parsed.get("title", ""),
            parsed.get("abstract", ""),
            parsed.get("sections", {}).get("introduction", ""),
        ])
        keywords = self.keyword_extractor.extract_keywords(analysis_text, top_k=8)

        job.current_step = f"summarizing {pdf_path.name}"
        job.logs.append(f"Generating summary for {pdf_path.name}")
        self.storage.save_job(job)
        summary_bundle = self.summarizer.summarize(parsed.get("cleaned_text", ""), keywords, self.summary_embedder)

        sections = parsed.get("sections", {})
        passages = []
        for section_name, section_text in sections.items():
            for chunk in chunk_passages(section_text):
                passages.append({"section": section_name, "text": chunk})
        if not passages and parsed.get("abstract"):
            passages = [{"section": "abstract", "text": parsed["abstract"]}]

        job.current_step = f"persisting {pdf_path.name}"
        job.logs.append(f"Saving metadata and text payload for {pdf_path.name}")
        self.storage.save_job(job)
        paper = PaperRecord(
            paper_id=paper_id,
            title=parsed.get("title", pdf_path.stem),
            authors=parsed.get("authors", []),
            year=parsed.get("year"),
            abstract=parsed.get("abstract", ""),
            keywords=keywords,
            summary=str(summary_bundle.get("summary", "")),
            sections=sections,
            pdf_path=str(copied_pdf_path),
            language=parsed.get("language", "unknown"),
            created_at=self.storage.now_iso(),
            important_sentences=list(summary_bundle.get("important_sentences", [])),
        )
        self.storage.upsert_paper(paper)
        self.storage.save_text_payload(
            paper_id,
            {
                "paper_id": paper_id,
                "title": paper.title,
                "raw_text": parsed.get("raw_text", ""),
                "cleaned_text": parsed.get("cleaned_text", ""),
                "passages": passages,
            },
        )
        job.logs.append(f"Stored paper {paper_id}")
        self.storage.save_job(job)

    def mark_interrupted_jobs(self) -> int:
        interrupted = 0
        for job in self.storage.list_jobs():
            if job.status != "running":
                continue
            job.status = "interrupted"
            job.current_step = "interrupted"
            job.logs.append(
                f"Import job was interrupted. Server PID {os.getpid()} started without an active worker for this job."
            )
            self.storage.save_job(job)
            interrupted += 1
        return interrupted
