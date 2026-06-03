from __future__ import annotations

from typing import Dict, List

import numpy as np

from .utils import sentence_split, tokenize


class Summarizer:
    def summarize(self, text: str, keywords: List[str], embedder, max_sentences: int = 4) -> Dict[str, List[str] | str]:
        sentences = sentence_split(text)
        if not sentences:
            return {"summary": "", "important_sentences": []}

        selected_sentences = sentences[: min(len(sentences), 24)]
        sentence_embeddings = embedder.encode(selected_sentences)
        document_embedding = embedder.encode_one(" ".join(selected_sentences))
        keyword_set = {kw.lower() for kw in keywords}

        scores = []
        total_sentences = max(len(selected_sentences), 1)
        for index, sentence in enumerate(selected_sentences):
            sentence_embedding = sentence_embeddings[index]
            sim = float(np.dot(sentence_embedding, document_embedding))
            sentence_tokens = {tok.lower() for tok in tokenize(sentence)}
            coverage = 0.0
            if keyword_set:
                coverage = len(sentence_tokens & keyword_set) / len(keyword_set)
            position_weight = 1.0 - (index / total_sentences)
            score = 0.5 * sim + 0.3 * coverage + 0.2 * position_weight
            scores.append((index, sentence, score))

        scores.sort(key=lambda item: item[2], reverse=True)
        top = sorted(scores[:max_sentences], key=lambda item: item[0])
        important = [sentence for _, sentence, _ in top]
        return {
            "summary": " ".join(important),
            "important_sentences": important,
        }

