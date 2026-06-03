from __future__ import annotations

import hashlib
import os
from typing import Iterable, List

import numpy as np

from .utils import tokenize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None


class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self.backend_name = "hashing"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            return
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            self._model = SentenceTransformer(self.model_name, local_files_only=True, device="cpu")
            self.backend_name = self.model_name
        except Exception:
            self._model = None

    @staticmethod
    def is_sentence_transformers_available() -> bool:
        return SentenceTransformer is not None

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._load_model()
        if self._model is not None:
            embeddings = self._model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        return np.vstack([self._hash_embed(text) for text in text_list]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 2):
                bucket = int.from_bytes(digest[offset:offset + 2], "little") % self.dim
                vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
