from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QApplication
from qfluentwidgets import (
    FluentIcon as FIF,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    NavigationItemPosition,
)

from .api_client import LitMindApiClient
from .pages.detail_page import DetailPage
from .pages.import_page import ImportPage
from .pages.library_page import LibraryPage
from .pages.search_page import SearchPage
from .workers import ApiWorker, JobPoller


class LitMindWindow(FluentWindow):
    def __init__(self) -> None:
        super().__init__()
        self.api = LitMindApiClient()
        self._active_detail_paper_id = ""
        self._workers = []
        self._poller = None
        self._build_pages()
        self._configure_window()
        self._connect_signals()
        self._load_library()

    def _build_pages(self) -> None:
        self.import_page = ImportPage(self)
        self.library_page = LibraryPage(self)
        self.search_page = SearchPage(self)
        self.detail_page = DetailPage(self)

        self.addSubInterface(self.import_page, FIF.DOWN, "导入", NavigationItemPosition.SCROLL)
        self.addSubInterface(self.library_page, FIF.LIBRARY, "文献库", NavigationItemPosition.SCROLL)
        self.addSubInterface(self.search_page, FIF.SEARCH, "智能检索", NavigationItemPosition.SCROLL)
        self.addSubInterface(self.detail_page, FIF.DOCUMENT, "详情", NavigationItemPosition.SCROLL)

    def _configure_window(self) -> None:
        self.resize(1200, 820)
        self.setMinimumWidth(900)
        self.setWindowTitle("LitMind 智能文献管理系统")
        desktop = QApplication.desktop().availableGeometry()
        self.move(desktop.width() // 2 - self.width() // 2, desktop.height() // 2 - self.height() // 2)

    def _connect_signals(self) -> None:
        self.import_page.import_requested.connect(self._start_import)
        self.library_page.refresh_requested.connect(self._load_library)
        self.library_page.paper_selected.connect(self._open_paper)
        self.search_page.search_requested.connect(self._run_search)
        self.search_page.paper_selected.connect(self._open_paper)
        self.detail_page.recommendation_selected.connect(self._open_paper)

    def _track_worker(self, worker) -> None:
        self._workers.append(worker)
        worker.finished.connect(lambda *_: self._cleanup_worker(worker))
        worker.failed.connect(lambda *_: self._cleanup_worker(worker))
        worker.start()

    def _cleanup_worker(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _notify_error(self, message: str) -> None:
        InfoBar.error(
            title="请求失败",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000,
        )

    def _notify_success(self, title: str, message: str) -> None:
        InfoBar.success(
            title=title,
            content=message,
            parent=self,
            position=InfoBarPosition.TOP,
            duration=2500,
        )

    def _start_import(self, payload: dict) -> None:
        directory = payload.get("directory", "")
        enable_online_metadata = payload.get("enable_online_metadata", False)
        local_path = Path(directory).expanduser()
        import_fn = self.api.import_local_directory if local_path.exists() and local_path.is_dir() else self.api.import_directory
        worker = ApiWorker(import_fn, directory, enable_online_metadata)
        worker.finished.connect(self._on_import_started)
        worker.failed.connect(self._notify_error)
        self._track_worker(worker)

    def _on_import_started(self, payload: dict) -> None:
        job = payload.get("data", {})
        self._notify_success("导入任务已启动", f"任务 ID: {job.get('job_id', '')}")
        if self._poller is not None:
            self._poller.stop()
            self._poller.deleteLater()
        self._poller = JobPoller(self.api, job.get("job_id", ""))
        self._poller.updated.connect(self._on_job_update)
        self._poller.failed.connect(self._notify_error)
        self._poller.start()

    def _on_job_update(self, payload: dict) -> None:
        self.import_page.set_job_payload(payload)
        status = payload.get("data", {}).get("status")
        if status in {"completed", "completed_with_errors"}:
            self._notify_success("导入完成", "文献解析和索引构建已完成")
            self._load_library()

    def _load_library(self, filters: dict | None = None) -> None:
        filters = filters or {}
        worker = ApiWorker(
            self.api.list_papers,
            filters.get("keyword", ""),
            filters.get("year", ""),
            filters.get("topic", ""),
        )
        worker.finished.connect(lambda payload: self.library_page.set_papers(payload.get("data", [])))
        worker.failed.connect(self._notify_error)
        self._track_worker(worker)

    def _run_search(self, query: str) -> None:
        worker = ApiWorker(self.api.search, query, 5)
        worker.finished.connect(lambda payload: self.search_page.set_results(payload.get("data", [])))
        worker.failed.connect(self._notify_error)
        self._track_worker(worker)

    def _open_paper(self, paper_id: str) -> None:
        self._active_detail_paper_id = paper_id
        self.detail_page.prepare_for_paper(paper_id)
        self.switchTo(self.detail_page)

        worker = ApiWorker(self.api.get_paper, paper_id)
        worker.finished.connect(lambda payload, pid=paper_id: self._on_paper_loaded(pid, payload))
        worker.failed.connect(self._notify_error)
        self._track_worker(worker)

        pdf_worker = ApiWorker(self.api.download_pdf, paper_id)
        pdf_worker.finished.connect(lambda pdf_bytes, pid=paper_id: self._on_pdf_loaded(pid, pdf_bytes))
        pdf_worker.failed.connect(lambda message, pid=paper_id: self._on_pdf_failed(pid, message))
        self._track_worker(pdf_worker)

        rec_worker = ApiWorker(self.api.recommendations, paper_id, 5)
        rec_worker.finished.connect(lambda payload, pid=paper_id: self._on_recommendations_loaded(pid, payload))
        rec_worker.failed.connect(self._notify_error)
        self._track_worker(rec_worker)

    def _on_paper_loaded(self, paper_id: str, payload: dict) -> None:
        if paper_id != self._active_detail_paper_id:
            return
        self.detail_page.set_paper(payload.get("data", {}))

    def _on_recommendations_loaded(self, paper_id: str, payload: dict) -> None:
        if paper_id != self._active_detail_paper_id:
            return
        self.detail_page.set_recommendations(payload.get("data", []))

    def _on_pdf_loaded(self, paper_id: str, pdf_bytes: bytes) -> None:
        if paper_id != self._active_detail_paper_id:
            return
        self.detail_page.set_pdf_bytes(pdf_bytes)

    def _on_pdf_failed(self, paper_id: str, message: str) -> None:
        if paper_id != self._active_detail_paper_id:
            return
        self.detail_page.set_pdf_error(message)
