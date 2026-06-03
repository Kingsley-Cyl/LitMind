from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server_ubuntu.server.services.utils import clean_abstract_text, chunk_passages, detect_language, sentence_split, tokenize


class UtilsTestCase(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("这是一个中文摘要"), "zh")
        self.assertEqual(detect_language("This is an English abstract."), "en")

    def test_sentence_split(self):
        sentences = sentence_split("Sentence one. Sentence two! Sentence three?")
        self.assertEqual(len(sentences), 3)

    def test_tokenize(self):
        tokens = tokenize("This paper studies semantic retrieval for literature recommendation.")
        self.assertIn("semantic", tokens)
        self.assertIn("retrieval", tokens)

    def test_chunk_passages(self):
        text = " ".join([f"Sentence {idx}." for idx in range(30)])
        chunks = chunk_passages(text, max_chars=80)
        self.assertGreater(len(chunks), 1)

    def test_clean_abstract_text(self):
        text = """
        Abstract
        Article Info
        Received 1 January 2026
        This paper proposes a retrieval method for literature analysis.
        Keywords: retrieval, literature
        """
        cleaned = clean_abstract_text(text)
        self.assertIn("This paper proposes a retrieval method", cleaned)
        self.assertNotIn("Article Info", cleaned)
        self.assertNotIn("Keywords", cleaned)


if __name__ == "__main__":
    unittest.main()
