from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class ClientSettings:
    base_url: str = os.environ.get("LITMIND_API_BASE_URL", "http://127.0.0.1:9000")
    request_timeout: int = 120


settings = ClientSettings()

