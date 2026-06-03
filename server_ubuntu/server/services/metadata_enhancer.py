from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests

from .utils import clean_abstract_text, normalize_whitespace


DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


class MetadataEnhancer:
    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "LitMind/0.1 (course-project metadata enhancer)",
                "Accept": "application/json",
            }
        )

    def enhance(self, parsed: Dict[str, object]) -> Tuple[Dict[str, object], Optional[str]]:
        raw_text = str(parsed.get("raw_text", ""))
        title = str(parsed.get("title", ""))
        doi = self.extract_doi(raw_text) or self.extract_doi(title)

        record = None
        source = None
        if doi:
            record = self.lookup_by_doi(doi)
            source = f"crossref:doi:{doi}" if record else None
        if record is None and title:
            record = self.lookup_by_title(title)
            source = "crossref:title" if record else None
        if record is None:
            return parsed, None

        updated = dict(parsed)
        if record.get("title"):
            updated["title"] = record["title"]
        if record.get("authors"):
            updated["authors"] = record["authors"]
        if record.get("year"):
            updated["year"] = record["year"]
        if record.get("doi"):
            updated["doi"] = record["doi"]
        if record.get("abstract") and self._should_replace_abstract(str(updated.get("abstract", ""))):
            updated["abstract"] = record["abstract"]
        updated["metadata_source"] = source
        return updated, source

    def extract_doi(self, text: str) -> Optional[str]:
        match = DOI_PATTERN.search(text or "")
        if not match:
            return None
        return match.group(0).rstrip(").,;")

    def lookup_by_doi(self, doi: str) -> Optional[Dict[str, object]]:
        try:
            response = self.session.get(
                f"https://api.crossref.org/works/{doi}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json().get("message", {})
            return self._normalize_crossref_record(payload)
        except Exception:
            return None

    def lookup_by_title(self, title: str) -> Optional[Dict[str, object]]:
        try:
            response = self.session.get(
                "https://api.crossref.org/works",
                params={
                    "query.title": title,
                    "rows": 5,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
            best = None
            best_score = 0.0
            normalized_title = self._normalize_title(title)
            for item in items:
                candidate_title = self._safe_first(item.get("title", []))
                if not candidate_title:
                    continue
                score = SequenceMatcher(None, normalized_title, self._normalize_title(candidate_title)).ratio()
                if score > best_score:
                    best = item
                    best_score = score
            if best is None or best_score < 0.55:
                return None
            return self._normalize_crossref_record(best)
        except Exception:
            return None

    def _normalize_crossref_record(self, item: Dict[str, object]) -> Dict[str, object]:
        title = normalize_whitespace(self._safe_first(item.get("title", [])))
        authors: List[str] = []
        for author in item.get("author", []) or []:
            given = normalize_whitespace(str(author.get("given", "")))
            family = normalize_whitespace(str(author.get("family", "")))
            if given or family:
                authors.append(normalize_whitespace(f"{given} {family}".strip()))

        year = None
        for key in ("published-print", "published-online", "issued", "created"):
            date_parts = (item.get(key) or {}).get("date-parts", [])
            if date_parts and date_parts[0]:
                try:
                    year = int(date_parts[0][0])
                    break
                except Exception:
                    continue

        abstract = clean_abstract_text(re.sub(r"<[^>]+>", " ", str(item.get("abstract", ""))))

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": normalize_whitespace(str(item.get("DOI", ""))),
        }

    def _should_replace_abstract(self, current_abstract: str) -> bool:
        text = current_abstract.lower()
        if not current_abstract.strip():
            return True
        return any(
            token in text
            for token in (
                "contents lists available",
                "journal homepage",
                "received ",
                "available online",
                "copyright",
            )
        )

    def _normalize_title(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _safe_first(self, values) -> str:
        if isinstance(values, list) and values:
            return str(values[0])
        return str(values or "")
