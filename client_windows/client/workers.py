from __future__ import annotations

from typing import Any, Callable

from PyQt5.QtCore import QThread, pyqtSignal


class ApiWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class JobPoller(QThread):
    updated = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, api_client, job_id: str, interval_ms: int = 1200) -> None:
        super().__init__()
        self.api_client = api_client
        self.job_id = job_id
        self.interval_ms = interval_ms
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            try:
                payload = self.api_client.get_job(self.job_id)
                self.updated.emit(payload)
                status = payload.get("data", {}).get("status")
                if status in {"completed", "completed_with_errors", "failed", "interrupted"}:
                    return
                self.msleep(self.interval_ms)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
