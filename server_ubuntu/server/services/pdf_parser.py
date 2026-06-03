from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import clean_abstract_text, clean_academic_text, detect_language, normalize_whitespace

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency at runtime
    fitz = None


SECTION_PATTERNS = {
    "abstract": r"^\s*(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\s*$",
    "introduction": r"^\s*(?:\d+(?:\.\d+)?\.?\s+)?introduction\s*$",
    "method": r"^\s*(?:\d+(?:\.\d+)?\.?\s+)?(?:method|methods|approach|methodology|proposed method)\s*$",
    "experiments": r"^\s*(?:\d+(?:\.\d+)?\.?\s+)?(?:experiment|experiments|evaluation|results)\s*$",
    "conclusion": r"^\s*(?:\d+(?:\.\d+)?\.?\s+)?(?:conclusion|conclusions)\s*$",
    "references": r"^\s*references\s*$",
}

BOILERPLATE_PATTERNS = [
    r"article info",
    r"contents lists available",
    r"journal homepage",
    r"science direct",
    r"available online",
    r"received .* revised",
    r"accepted .*",
    r"https?://doi\.org/",
    r"all rights reserved",
    r"text and data mining",
    r"corresponding author",
    r"e-mail address",
]

AFFILIATION_HINTS = re.compile(
    r"\b(university|institute|school|college|department|laboratory|lab|centre|center|ministry|hospital|academy)\b",
    re.I,
)

NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){0,3}\b")


class PDFParser:
    @staticmethod
    def is_available() -> bool:
        return fitz is not None

    def parse_paper(self, pdf_path: str | Path) -> Dict[str, object]:
        layout = self.extract_layout(pdf_path)
        raw_text = self.reconstruct_document_text(layout)
        cleaned = clean_academic_text(self.remove_boilerplate(raw_text))
        metadata = self.extract_metadata(layout, cleaned, Path(pdf_path).stem)
        metadata["raw_text"] = raw_text
        metadata["cleaned_text"] = cleaned
        return metadata

    def extract_layout(self, pdf_path: str | Path) -> Dict[str, object]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is not installed. Please install PyMuPDF to parse PDFs.")

        pdf = fitz.open(str(pdf_path))
        pages: List[Dict[str, object]] = []
        for page_number, page in enumerate(pdf):
            page_dict = page.get_text("dict", sort=True)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            blocks = []
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                text, max_font, line_count = self._extract_block_text(block)
                text = normalize_whitespace(text)
                if not text:
                    continue
                bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                blocks.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "font_size": max_font,
                        "line_count": line_count,
                        "page_number": page_number,
                    }
                )
            pages.append(
                {
                    "width": page_width,
                    "height": page_height,
                    "blocks": blocks,
                    "page_number": page_number,
                }
            )
        pdf.close()
        return {"pages": pages}

    def reconstruct_document_text(self, layout: Dict[str, object]) -> str:
        pages = layout.get("pages", [])
        page_texts = []
        for page in pages:
            ordered = self._order_blocks(page.get("blocks", []), page.get("width", 0.0))
            page_texts.append("\n\n".join(block["text"] for block in ordered))
        return "\n\n".join(page_texts)

    def extract_metadata(self, layout: Dict[str, object], cleaned_text: str, source_name: str) -> Dict[str, object]:
        pages = layout.get("pages", [])
        first_page = pages[0] if pages else {"blocks": [], "height": 0.0, "width": 0.0}
        ordered_first_blocks = self._order_blocks(first_page.get("blocks", []), first_page.get("width", 0.0))
        page_text = "\n\n".join(block["text"] for block in ordered_first_blocks)
        normalized_page_text = self._normalize_headings(page_text)

        title, title_block = self._extract_title(ordered_first_blocks, source_name, first_page.get("height", 0.0))
        authors = self._extract_authors(ordered_first_blocks, title_block, first_page.get("height", 0.0))
        year = self._extract_year(cleaned_text)
        abstract = self._extract_abstract(ordered_first_blocks, normalized_page_text)
        sections = self.split_sections(cleaned_text)
        if abstract and "abstract" not in sections:
            sections["abstract"] = abstract

        return {
            "title": normalize_whitespace(title),
            "authors": authors,
            "year": year,
            "abstract": normalize_whitespace(abstract),
            "sections": sections,
            "language": detect_language(cleaned_text),
        }

    def split_sections(self, cleaned_text: str) -> Dict[str, str]:
        text = self._normalize_headings(cleaned_text)
        lines = text.splitlines()
        matches: List[Tuple[str, int]] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            for name, pattern in SECTION_PATTERNS.items():
                if re.match(pattern, stripped, re.I):
                    matches.append((name, index))
                    break

        if not matches:
            return {}

        matches.sort(key=lambda item: item[1])
        sections: Dict[str, str] = {}
        for idx, (name, start_line) in enumerate(matches):
            end_line = matches[idx + 1][1] if idx + 1 < len(matches) else len(lines)
            content = "\n".join(lines[start_line + 1:end_line]).strip()
            content = self.remove_boilerplate(content)
            if content:
                sections[name] = normalize_whitespace(content)
        return sections

    def remove_boilerplate(self, text: str) -> str:
        lines = text.splitlines()
        kept = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            lowered = stripped.lower()
            if any(re.search(pattern, lowered, re.I) for pattern in BOILERPLATE_PATTERNS):
                continue
            kept.append(stripped)
        return normalize_whitespace("\n".join(kept))

    def _extract_block_text(self, block: Dict[str, object]) -> Tuple[str, float, int]:
        lines = []
        max_font = 0.0
        line_count = 0
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                raw = str(span.get("text", ""))
                if raw.strip():
                    spans.append(raw)
                max_font = max(max_font, float(span.get("size", 0.0)))
            if spans:
                lines.append(" ".join(spans))
                line_count += 1
        return "\n".join(lines), max_font, line_count

    def _order_blocks(self, blocks: List[Dict[str, object]], page_width: float) -> List[Dict[str, object]]:
        if not blocks:
            return []
        text_blocks = [block for block in blocks if block.get("text", "").strip()]
        left_candidates = [block for block in text_blocks if block["bbox"][0] < page_width * 0.45]
        right_candidates = [block for block in text_blocks if block["bbox"][0] > page_width * 0.55]
        two_columns = len(left_candidates) >= 3 and len(right_candidates) >= 3
        if not two_columns:
            return sorted(text_blocks, key=lambda item: (round(item["bbox"][1], 1), round(item["bbox"][0], 1)))

        left = [block for block in text_blocks if block["bbox"][0] < page_width * 0.52]
        right = [block for block in text_blocks if block["bbox"][0] >= page_width * 0.52]
        left.sort(key=lambda item: (round(item["bbox"][1], 1), round(item["bbox"][0], 1)))
        right.sort(key=lambda item: (round(item["bbox"][1], 1), round(item["bbox"][0], 1)))
        return left + right

    def _extract_title(
        self,
        first_page_blocks: List[Dict[str, object]],
        source_name: str,
        page_height: float,
    ) -> Tuple[str, Optional[Dict[str, object]]]:
        candidate_blocks = []
        for block in first_page_blocks:
            text = block["text"]
            lowered = text.lower()
            if any(re.search(pattern, lowered, re.I) for pattern in BOILERPLATE_PATTERNS):
                continue
            if AFFILIATION_HINTS.search(text):
                continue
            if "@" in text or len(text) < 12:
                continue
            if block["bbox"][1] > page_height * 0.45:
                continue
            score = block["font_size"] * 4 - block["bbox"][1] * 0.002 + min(len(text), 140) * 0.02
            candidate_blocks.append((score, block))

        if not candidate_blocks:
            return source_name, None

        candidate_blocks.sort(key=lambda item: item[0], reverse=True)
        title_block = candidate_blocks[0][1]
        title = self._normalize_title_text(title_block["text"])
        return title or source_name, title_block

    def _extract_authors(
        self,
        first_page_blocks: List[Dict[str, object]],
        title_block: Optional[Dict[str, object]],
        page_height: float,
    ) -> List[str]:
        if not first_page_blocks:
            return []
        title_bottom = title_block["bbox"][3] if title_block else 0.0
        abstract_top = self._find_heading_top(first_page_blocks, ("abstract", "a b s t r a c t", "a b s t r a c t"))
        abstract_limit = abstract_top if abstract_top is not None else page_height * 0.42

        names: List[str] = []
        seen = set()
        for block in first_page_blocks:
            y0, y1 = block["bbox"][1], block["bbox"][3]
            text = block["text"]
            if y1 <= title_bottom or y0 >= abstract_limit:
                continue
            if AFFILIATION_HINTS.search(text) or "@" in text or len(text) > 220:
                continue
            cleaned = re.sub(r"[\d,*†‡§#∗]+", " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned)
            for candidate in NAME_PATTERN.findall(cleaned):
                lowered = candidate.lower()
                if lowered in seen:
                    continue
                if any(token in lowered for token in ("journal", "pattern", "recognition", "available", "abstract")):
                    continue
                seen.add(lowered)
                names.append(candidate)
        return names[:8]

    def _extract_abstract(self, first_page_blocks: List[Dict[str, object]], normalized_page_text: str) -> str:
        heading_top = self._find_heading_top(first_page_blocks, ("abstract", "a b s t r a c t"))
        intro_top = self._find_heading_top(first_page_blocks, ("introduction", "1. introduction", "1 introduction"))
        if heading_top is not None:
            abstract_chunks = []
            for block in first_page_blocks:
                y0 = block["bbox"][1]
                if y0 <= heading_top:
                    continue
                if intro_top is not None and y0 >= intro_top:
                    break
                text = block["text"]
                if any(re.search(pattern, text, re.I) for pattern in SECTION_PATTERNS.values()):
                    continue
                if re.match(r"^\s*(?:keywords|index terms|key words)\s*:?", text, re.I):
                    break
                abstract_chunks.append(text)
            abstract_text = clean_abstract_text("\n".join(abstract_chunks))
            abstract_text = self.remove_boilerplate(abstract_text)
            if len(abstract_text) > 80:
                return abstract_text

        return self._extract_abstract_fallback(normalized_page_text)

    def _extract_year(self, cleaned_text: str) -> Optional[int]:
        years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", cleaned_text[:5000])]
        if not years:
            return None
        plausible = [year for year in years if 1990 <= year <= 2100]
        return max(plausible) if plausible else years[0]

    def _extract_abstract_fallback(self, cleaned_text: str) -> str:
        text = self._normalize_headings(cleaned_text)
        match = re.search(
            r"(?:^|\n)\s*abstract\s*\n([\s\S]{80,3000}?)(?=\n\s*(?:keywords|index terms|1\.?\s*introduction|introduction)\b)",
            text,
            re.I,
        )
        if match:
            return clean_abstract_text(match.group(1).strip())
        first_chunk = text[:1800]
        sentences = re.split(r"(?<=[.!?。！？])\s+", first_chunk)
        return clean_abstract_text(" ".join(sentences[:5]).strip())

    def _find_heading_top(self, blocks: List[Dict[str, object]], keywords: Tuple[str, ...]) -> Optional[float]:
        for block in blocks:
            normalized = self._normalize_headings(block["text"]).strip().lower()
            if normalized in {keyword.lower() for keyword in keywords}:
                return float(block["bbox"][1])
            if any(normalized.startswith(keyword.lower()) for keyword in keywords):
                return float(block["bbox"][1])
        return None

    def _normalize_title_text(self, text: str) -> str:
        text = normalize_whitespace(text.replace("\n", " "))
        text = re.sub(r"\s+", " ", text)
        return self.remove_boilerplate(text)

    def _normalize_headings(self, text: str) -> str:
        replacements = {
            r"\ba\s*b\s*s\s*t\s*r\s*a\s*c\s*t\b": "abstract",
            r"\bk\s*e\s*y\s*w\s*o\s*r\s*d\s*s\b": "keywords",
            r"\br\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s\b": "references",
        }
        normalized = text
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.I)
        return normalized
