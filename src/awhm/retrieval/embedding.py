"""Embedding services: sentence-transformers wrapper plus a deterministic mock."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class EmbeddingService(Protocol):
    """Protocol for embedding services (allows easy mocking)."""

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_single(self, text: str) -> np.ndarray: ...

    @property
    def dim(self) -> int: ...


class SentenceTransformerEmbedding:
    """Embedding service backed by sentence-transformers.

    The model is loaded lazily on first use so that constructing a session
    stays cheap and the optional dependency is only required when embeddings
    are actually computed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "sentence-transformers is not installed. Install it with "
                    "`pip install 'awhm-lite[embeddings]'`, or start the session "
                    "with use_mock_embeddings=True."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class MockEmbeddingService:
    """Fast deterministic embeddings for tests and offline runs.

    Vectors are seeded from a SHA-256 of the text, so the same text always
    maps to the same vector across processes (Python's built-in ``hash`` is
    salted per process and would not).
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            result[i] = vec / (np.linalg.norm(vec) + 1e-8)
        return result

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of ``matrix`` with each row scaled to unit length."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of ``matrix``."""
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    return normalize_rows(matrix) @ q_norm


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for every row of ``a`` against every row of ``b``.

    Returns a matrix of shape ``(len(a), len(b))``.
    """
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0] if a.ndim == 2 else 0, b.shape[0] if b.ndim == 2 else 0),
                        dtype=np.float32)
    return normalize_rows(a) @ normalize_rows(b).T
