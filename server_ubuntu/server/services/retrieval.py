from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import (
    DOC_EMBEDDINGS_FILE,
    DOC_INDEX_FILE,
    DOC_INDEX_META_FILE,
    PASSAGE_EMBEDDINGS_FILE,
    PASSAGE_INDEX_FILE,
    PASSAGE_INDEX_META_FILE,
    RECOMMEND_DOC_EMBEDDINGS_FILE,
    RECOMMEND_DOC_INDEX_FILE,
    RECOMMEND_DOC_INDEX_META_FILE,
)
from .utils import read_json, tokenize, write_json

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency at runtime
    faiss = None


class RetrievalEngine:
    def __init__(self, search_embedder, recommend_embedder=None, dense_weight: float = 0.5) -> None:
        self.search_embedder = search_embedder
        self.recommend_embedder = recommend_embedder or search_embedder
        self.dense_weight = dense_weight

        self.doc_embeddings = np.zeros((0, self.search_embedder.dim), dtype=np.float32)
        self.passage_embeddings = np.zeros((0, self.search_embedder.dim), dtype=np.float32)
        self.recommend_embeddings = np.zeros((0, self.recommend_embedder.dim), dtype=np.float32)

        self.doc_meta: List[Dict] = []
        self.passage_meta: List[Dict] = []
        self.recommend_meta: List[Dict] = []

        self.doc_index = None
        self.passage_index = None
        self.recommend_index = None

        self.doc_sparse = self._empty_sparse_index()
        self.passage_sparse = self._empty_sparse_index()
        self.recommend_sparse = self._empty_sparse_index()

        self.backend_name = "hybrid"
        self.load()

    def load(self) -> None:
        if DOC_EMBEDDINGS_FILE.exists():
            self.doc_embeddings = np.load(DOC_EMBEDDINGS_FILE)
        if PASSAGE_EMBEDDINGS_FILE.exists():
            self.passage_embeddings = np.load(PASSAGE_EMBEDDINGS_FILE)
        if RECOMMEND_DOC_EMBEDDINGS_FILE.exists():
            self.recommend_embeddings = np.load(RECOMMEND_DOC_EMBEDDINGS_FILE)

        self.doc_meta = read_json(DOC_INDEX_META_FILE, default=[])
        self.passage_meta = read_json(PASSAGE_INDEX_META_FILE, default=[])
        self.recommend_meta = read_json(RECOMMEND_DOC_INDEX_META_FILE, default=[])
        if not self.recommend_meta:
            self.recommend_meta = list(self.doc_meta)

        if faiss is not None and DOC_INDEX_FILE.exists() and PASSAGE_INDEX_FILE.exists():
            self.doc_index = faiss.read_index(str(DOC_INDEX_FILE))
            self.passage_index = faiss.read_index(str(PASSAGE_INDEX_FILE))
        else:
            self.doc_index = None
            self.passage_index = None
        if faiss is not None and RECOMMEND_DOC_INDEX_FILE.exists():
            self.recommend_index = faiss.read_index(str(RECOMMEND_DOC_INDEX_FILE))
        else:
            self.recommend_index = None

        self.doc_sparse = self._fit_sparse_index([item.get("search_text", item.get("title", "")) for item in self.doc_meta])
        self.passage_sparse = self._fit_sparse_index([item.get("text", "") for item in self.passage_meta])
        self.recommend_sparse = self._fit_sparse_index([item.get("recommend_text", item.get("search_text", item.get("title", ""))) for item in self.recommend_meta])
        self.backend_name = "hybrid"

    def rebuild(self, papers: List[Dict], payloads: Dict[str, Dict]) -> None:
        doc_meta: List[Dict] = []
        passage_meta: List[Dict] = []
        recommend_meta: List[Dict] = []
        doc_vectors: List[np.ndarray] = []
        passage_vectors: List[np.ndarray] = []
        recommend_vectors: List[np.ndarray] = []

        for paper in papers:
            payload = payloads.get(paper["paper_id"], {})
            cleaned_text = payload.get("cleaned_text", "").strip()
            doc_text = " ".join([
                paper["title"],
                paper.get("abstract", ""),
                " ".join(paper.get("keywords", [])),
                cleaned_text,
            ]).strip()
            doc_meta.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "search_text": doc_text,
                }
            )
            recommend_meta.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "recommend_text": doc_text,
                }
            )
            doc_vectors.append(self.search_embedder.encode_one(doc_text))
            recommend_vectors.append(self.recommend_embedder.encode_one(doc_text))
            for passage in payload.get("passages", []):
                passage_text = passage.get("text", "").strip()
                if not passage_text:
                    continue
                passage_meta.append(
                    {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "section": passage.get("section", ""),
                        "text": passage_text,
                    }
                )
                passage_vectors.append(self.search_embedder.encode_one(passage_text))

        self.doc_meta = doc_meta
        self.passage_meta = passage_meta
        self.recommend_meta = recommend_meta
        self.doc_embeddings = np.vstack(doc_vectors).astype(np.float32) if doc_vectors else np.zeros((0, self.search_embedder.dim), dtype=np.float32)
        self.passage_embeddings = (
            np.vstack(passage_vectors).astype(np.float32)
            if passage_vectors else np.zeros((0, self.search_embedder.dim), dtype=np.float32)
        )
        self.recommend_embeddings = (
            np.vstack(recommend_vectors).astype(np.float32)
            if recommend_vectors else np.zeros((0, self.recommend_embedder.dim), dtype=np.float32)
        )

        self.doc_sparse = self._fit_sparse_index([item["search_text"] for item in self.doc_meta])
        self.passage_sparse = self._fit_sparse_index([item["text"] for item in self.passage_meta])
        self.recommend_sparse = self._fit_sparse_index([item["recommend_text"] for item in self.recommend_meta])

        np.save(DOC_EMBEDDINGS_FILE, self.doc_embeddings)
        np.save(PASSAGE_EMBEDDINGS_FILE, self.passage_embeddings)
        np.save(RECOMMEND_DOC_EMBEDDINGS_FILE, self.recommend_embeddings)
        write_json(DOC_INDEX_META_FILE, self.doc_meta)
        write_json(PASSAGE_INDEX_META_FILE, self.passage_meta)
        write_json(RECOMMEND_DOC_INDEX_META_FILE, self.recommend_meta)

        if faiss is not None:
            self.doc_index = self._build_faiss_index(self.doc_embeddings)
            self.passage_index = self._build_faiss_index(self.passage_embeddings)
            self.recommend_index = self._build_faiss_index(self.recommend_embeddings)
            faiss.write_index(self.doc_index, str(DOC_INDEX_FILE))
            faiss.write_index(self.passage_index, str(PASSAGE_INDEX_FILE))
            faiss.write_index(self.recommend_index, str(RECOMMEND_DOC_INDEX_FILE))
        else:
            self.doc_index = None
            self.passage_index = None
            self.recommend_index = None
        self.backend_name = "hybrid"

    def search_documents(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if len(self.doc_meta) == 0:
            return []
        dense_scores = self._dense_scores(self.search_embedder.encode_one(query), self.doc_embeddings)
        sparse_scores = self._bm25_scores(query, self.doc_sparse)
        combined_scores = self._combine_scores(dense_scores, sparse_scores)
        return self._rank_scores(combined_scores, self.doc_meta, top_k)

    def search_passages(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if len(self.passage_meta) == 0:
            return []
        dense_scores = self._dense_scores(self.search_embedder.encode_one(query), self.passage_embeddings)
        sparse_scores = self._bm25_scores(query, self.passage_sparse)
        combined_scores = self._combine_scores(dense_scores, sparse_scores)
        return self._rank_scores(combined_scores, self.passage_meta, top_k)

    def recommend(self, paper_id: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        ids = [item["paper_id"] for item in self.recommend_meta]
        if paper_id not in ids:
            return []
        index = ids.index(paper_id)
        target_embedding = self.recommend_embeddings[index] if len(self.recommend_embeddings) == len(self.recommend_meta) else self.doc_embeddings[index]
        target_text = self.recommend_meta[index].get("recommend_text", self.recommend_meta[index].get("title", ""))
        dense_scores = self._dense_scores(target_embedding, self.recommend_embeddings if len(self.recommend_embeddings) == len(self.recommend_meta) else self.doc_embeddings)
        sparse_scores = self._bm25_scores(target_text, self.recommend_sparse)
        combined_scores = self._combine_scores(dense_scores, sparse_scores)
        return self._rank_scores(combined_scores, self.recommend_meta, top_k, exclude_index=index)

    def _rank_scores(
        self,
        scores: np.ndarray,
        meta: List[Dict],
        top_k: int,
        exclude_index: int | None = None,
    ) -> List[Tuple[Dict, float]]:
        if len(meta) == 0 or len(scores) == 0:
            return []
        order = np.argsort(scores)[::-1]
        results = []
        for idx in order:
            if exclude_index is not None and int(idx) == exclude_index:
                continue
            results.append((meta[int(idx)], float(scores[int(idx)])))
            if len(results) >= top_k:
                break
        return results

    def _dense_scores(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.zeros(0, dtype=np.float32)
        return embeddings @ query_embedding

    def _combine_scores(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
        if len(dense_scores) == 0:
            return self._minmax(sparse_scores)
        if len(sparse_scores) == 0:
            return self._minmax(dense_scores)
        dense_scaled = self._zscore(dense_scores)
        sparse_scaled = self._zscore(sparse_scores)
        combined = self.dense_weight * dense_scaled + (1.0 - self.dense_weight) * sparse_scaled
        return self._minmax(combined)

    def _zscore(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - mean) / std).astype(np.float32)

    def _minmax(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        min_v = float(values.min())
        max_v = float(values.max())
        if max_v - min_v < 1e-8:
            return np.ones_like(values, dtype=np.float32)
        return ((values - min_v) / (max_v - min_v)).astype(np.float32)

    def _empty_sparse_index(self) -> Dict[str, object]:
        return {
            "term_freqs": [],
            "doc_lengths": [],
            "idf": {},
            "avg_doc_len": 0.0,
            "document_count": 0,
        }

    def _fit_sparse_index(self, texts: List[str]) -> Dict[str, object]:
        if not texts:
            return self._empty_sparse_index()
        doc_tokens = [tokenize(text) for text in texts]
        doc_freq: Counter[str] = Counter()
        term_freqs: List[Counter[str]] = []
        doc_lengths: List[int] = []
        for tokens in doc_tokens:
            freq = Counter(tokens)
            term_freqs.append(freq)
            doc_lengths.append(len(tokens))
            doc_freq.update(set(tokens))
        document_count = len(doc_tokens)
        avg_doc_len = (sum(doc_lengths) / document_count) if document_count else 0.0
        idf = {
            term: float(np.log((document_count - df + 0.5) / (df + 0.5) + 1.0))
            for term, df in doc_freq.items()
        }
        return {
            "term_freqs": term_freqs,
            "doc_lengths": doc_lengths,
            "idf": idf,
            "avg_doc_len": avg_doc_len,
            "document_count": document_count,
        }

    def _bm25_scores(self, query: str, sparse_index: Dict[str, object], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        document_count = int(sparse_index.get("document_count", 0))
        scores = np.zeros(document_count, dtype=np.float32)
        if document_count == 0:
            return scores
        query_counts = Counter(tokenize(query))
        if not query_counts:
            return scores

        term_freqs: List[Counter[str]] = sparse_index["term_freqs"]
        doc_lengths: List[int] = sparse_index["doc_lengths"]
        idf: Dict[str, float] = sparse_index["idf"]
        avg_doc_len = float(sparse_index.get("avg_doc_len", 0.0)) or 1.0

        for term, qf in query_counts.items():
            term_idf = idf.get(term)
            if term_idf is None:
                continue
            for idx, freq in enumerate(term_freqs):
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                dl = doc_lengths[idx] or 1
                denom = tf + k1 * (1.0 - b + b * (dl / avg_doc_len))
                scores[idx] += term_idf * (tf * (k1 + 1.0) / denom) * qf
        return scores

    def _build_faiss_index(self, embeddings: np.ndarray):
        if len(embeddings) == 0:
            return faiss.IndexFlatIP(embeddings.shape[1] if embeddings.ndim == 2 and embeddings.shape[1] > 0 else self.search_embedder.dim)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        return index
