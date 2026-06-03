from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "with",
    "we", "this", "these", "those", "our", "their", "than", "then", "into", "using",
    "use", "used", "based", "study", "paper", "method", "methods", "results",
}

CN_STOPWORDS = {
    "的", "了", "和", "是", "在", "与", "及", "对", "中", "为", "研究", "本文", "一种", "进行",
    "通过", "提出", "基于", "以及", "可以", "我们",
}

ABSTRACT_NOISE_PATTERNS = (
    r"^\s*abstract\s*[:.\-]?\s*$",
    r"^\s*article info\s*$",
    r"^\s*keywords?\s*:?\s*$",
    r"^\s*index terms?\s*:?\s*$",
    r"^\s*key words\s*:?\s*$",
    r"^\s*corresponding author\b.*$",
    r"^\s*e-?mail address\b.*$",
    r"^\s*received\b.*$",
    r"^\s*revised\b.*$",
    r"^\s*accepted\b.*$",
    r"^\s*available online\b.*$",
    r"^\s*https?://doi\.org/\S+\s*$",
)

ABSTRACT_STOP_PATTERNS = (
    r"^\s*keywords?\s*:",
    r"^\s*index terms?\s*:",
    r"^\s*key words\s*:",
)


def model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def clean_academic_text(text: str) -> str:
    lines = [line.strip() for line in normalize_whitespace(text).splitlines()]
    cleaned = []
    for line in lines:
        if not line:
            cleaned.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if len(line) < 3 and not re.search(r"[A-Za-z\u4e00-\u9fff]", line):
            continue
        cleaned.append(line)
    return normalize_whitespace("\n".join(cleaned))


def clean_abstract_text(text: str) -> str:
    lines = [line.strip() for line in normalize_whitespace(text).splitlines()]
    cleaned = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        lowered = line.lower()
        if any(re.match(pattern, lowered, re.I) for pattern in ABSTRACT_NOISE_PATTERNS):
            continue
        if any(re.match(pattern, lowered, re.I) for pattern in ABSTRACT_STOP_PATTERNS):
            break
        cleaned.append(line)

    text = normalize_whitespace("\n".join(cleaned))
    text = re.sub(r"^\s*abstract\s*[:.\-]?\s*", "", text, flags=re.I)
    text = re.sub(r"\barticle info\b", "", text, flags=re.I)
    text = re.sub(r"\s*\b(?:keywords?|index terms?|key words)\s*:\s*.*$", "", text, flags=re.I | re.S)
    return normalize_whitespace(text)


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if cjk_count > latin_count:
        return "zh"
    if latin_count > 0:
        return "en"
    return "unknown"


def sentence_split(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def tokenize(text: str) -> List[str]:
    english = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens = english + chinese
    return [tok for tok in tokens if tok not in EN_STOPWORDS and tok not in CN_STOPWORDS]


def top_terms_by_frequency(text: str, top_k: int = 8) -> List[str]:
    counts = Counter(tokenize(text))
    return [word for word, _ in counts.most_common(top_k)]


def cosine_similarity(a, b) -> float:
    denom = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
    if denom == 0.0:
        return 0.0
    return float((a @ b) / denom)


def chunk_passages(text: str, max_chars: int = 700) -> List[str]:
    passages: List[str] = []
    current = []
    current_len = 0
    for sentence in sentence_split(text):
        if current_len + len(sentence) > max_chars and current:
            passages.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
    if current:
        passages.append(" ".join(current))
    return passages


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]
