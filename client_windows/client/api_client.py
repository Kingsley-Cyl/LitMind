from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .settings import settings


class LitMindApiClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or settings.base_url).rstrip("/")

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def import_directory(self, directory: str, enable_online_metadata: bool = False) -> Dict[str, Any]:
        return self._post(
            "/papers/import",
            {
                "directory": directory,
                "enable_online_metadata": enable_online_metadata,
            },
        )

    def import_local_directory(self, directory: str, enable_online_metadata: bool = False) -> Dict[str, Any]:
        root = Path(directory).expanduser()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"目录不存在: {root}")
        pdf_paths = sorted(path for path in root.rglob("*.pdf") if path.is_file())
        if not pdf_paths:
            raise FileNotFoundError(f"目录中未找到 PDF 文件: {root}")

        files = []
        handles = []
        try:
            for pdf_path in pdf_paths:
                handle = pdf_path.open("rb")
                handles.append(handle)
                files.append(("files", (pdf_path.name, handle, "application/pdf")))
            response = requests.post(
                f"{self.base_url}/papers/import-upload",
                data={"enable_online_metadata": str(enable_online_metadata).lower()},
                files=files,
                timeout=max(settings.request_timeout, 600),
            )
            response.raise_for_status()
            return response.json()
        finally:
            for handle in handles:
                handle.close()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._get(f"/jobs/{job_id}")

    def list_papers(self, keyword: str = "", year: str = "", topic: str = "") -> Dict[str, Any]:
        params = {}
        if keyword:
            params["keyword"] = keyword
        if year:
            params["year"] = year
        if topic:
            params["topic"] = topic
        return self._get("/papers", params=params)

    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        return self._get(f"/papers/{paper_id}")

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self._post("/search", {"query": query, "top_k": top_k})

    def recommendations(self, paper_id: str, top_k: int = 5) -> Dict[str, Any]:
        return self._get(f"/papers/{paper_id}/recommendations", params={"top_k": top_k})

    def download_pdf(self, paper_id: str) -> bytes:
        response = requests.get(
            f"{self.base_url}/papers/{paper_id}/pdf",
            timeout=max(settings.request_timeout, 300),
        )
        response.raise_for_status()
        return response.content

    def rebuild_index(self) -> Dict[str, Any]:
        return self._post("/index/rebuild", {})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        return response.json()
