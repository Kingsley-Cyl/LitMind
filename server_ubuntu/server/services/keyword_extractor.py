from __future__ import annotations

from collections import Counter, defaultdict
from typing import List
import os

from .utils import tokenize, top_terms_by_frequency

try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency at runtime
    KeyBERT = None
    SentenceTransformer = None


class KeywordExtractor:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        self._keybert = None
        self.backend_name = "textrank"

    @staticmethod
    def is_keybert_available() -> bool:
        return KeyBERT is not None

    def extract_by_textrank(self, text: str, top_k: int = 8) -> List[str]:
        tokens = tokenize(text)
        if not tokens:
            return []

        graph = defaultdict(set)
        window_size = 4
        for idx, token in enumerate(tokens):
            for neighbor in tokens[idx + 1: idx + window_size]:
                graph[token].add(neighbor)
                graph[neighbor].add(token)

        scores = {token: 1.0 for token in graph}
        for _ in range(20):
            next_scores = {}
            for token, neighbors in graph.items():
                rank = 0.15
                for neighbor in neighbors:
                    degree = max(len(graph[neighbor]), 1)
                    rank += 0.85 * (scores[neighbor] / degree)
                next_scores[token] = rank
            scores = next_scores

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [token for token, _ in ranked[:top_k]]

    def extract_by_keybert(self, text: str, top_k: int = 8) -> List[str]:
        if KeyBERT is None:
            return []
        if self._keybert is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                if SentenceTransformer is not None:
                    encoder = SentenceTransformer(self.model_name, local_files_only=True, device="cpu")
                    self._keybert = KeyBERT(encoder)
                else:
                    self._keybert = KeyBERT()
            except Exception:
                self._keybert = KeyBERT()
            self.backend_name = "textrank+keybert"

        keywords = self._keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_k,
        )
        return [item[0] for item in keywords]

    def extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        textrank = self.extract_by_textrank(text, top_k=top_k)
        semantic = self.extract_by_keybert(text, top_k=top_k)
        merged: List[str] = []
        seen = set()
        for word in textrank + semantic + top_terms_by_frequency(text, top_k=top_k):
            key = word.lower()
            if key not in seen and len(word.strip()) > 1:
                seen.add(key)
                merged.append(word.strip())
            if len(merged) >= top_k:
                break
        return merged
