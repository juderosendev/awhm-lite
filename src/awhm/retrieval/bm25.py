"""BM25Index: rank-bm25 wrapper with simple tokenization + stopword removal."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# Small hardcoded stopword set (avoids NLTK dependency)
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "this", "that", "these",
    "those", "what", "which", "who", "whom",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase alphanumeric tokens, stopwords removed."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """BM25 index over a corpus of documents."""

    def __init__(self, documents: list[str] | None = None) -> None:
        self._documents: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if documents:
            self.build(documents)

    def build(self, documents: list[str]) -> None:
        """Build/rebuild the index from a list of document strings."""
        self._documents = documents
        self._tokenized = [tokenize(d) for d in documents]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search the index. Returns list of (doc_index, score) sorted by score desc.

        Uses relative scoring: normalizes against max score so that even
        small corpora (where IDF can be negative) return results.
        """
        if self._bm25 is None or not self._documents:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        max_score = float(max(scores)) if len(scores) > 0 else 0.0
        # If all scores are non-positive, shift so best score = 1.0
        if max_score <= 0 and len(scores) > 0:
            min_score = float(min(scores))
            shift = abs(min_score) + 1.0
            indexed = [(i, float(s) + shift) for i, s in enumerate(scores)]
        else:
            indexed = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    @property
    def document_count(self) -> int:
        return len(self._documents)
