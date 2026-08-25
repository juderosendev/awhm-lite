"""A small, dependency-free BM25 index.

Uses the Lucene formulation of IDF, ``ln(1 + (N - n + 0.5) / (n + 0.5))``,
which is always positive. The classic Robertson IDF goes negative for terms
that appear in more than half the documents, which on the tiny corpora a
personal memory graph starts with meant either no results or every document
scoring above zero after an ad-hoc shift.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Small hardcoded stopword set (avoids an NLTK dependency)
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
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """BM25 over a list of documents (Okapi weighting, Lucene IDF)."""

    def __init__(
        self,
        documents: list[str] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._documents: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_len: float = 0.0
        self._idf: dict[str, float] = {}
        if documents:
            self.build(documents)

    def build(self, documents: list[str]) -> None:
        """Build or rebuild the index from a list of document strings."""
        self._documents = list(documents)
        tokenized = [tokenize(d) for d in self._documents]
        self._term_freqs = [Counter(tokens) for tokens in tokenized]
        self._doc_lengths = [len(tokens) for tokens in tokenized]
        n_docs = len(self._documents)
        self._avg_len = (sum(self._doc_lengths) / n_docs) if n_docs else 0.0

        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        self._idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query: str) -> list[float]:
        """Raw BM25 score for every document, in index order."""
        q_tokens = tokenize(query)
        if not q_tokens or not self._documents:
            return [0.0] * len(self._documents)
        results: list[float] = []
        for tf, length in zip(self._term_freqs, self._doc_lengths, strict=True):
            norm = self.k1 * (1.0 - self.b + self.b * (length / self._avg_len if self._avg_len else 0.0))
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
            results.append(score)
        return results

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs with score > 0, best first."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def document_count(self) -> int:
        return len(self._documents)
