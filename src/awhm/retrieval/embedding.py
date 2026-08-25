"""EmbeddingService: sentence-transformers wrapper for encode/cosine similarity."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingService(Protocol):
    """Protocol for embedding services (allows easy mocking)."""

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_single(self, text: str) -> np.ndarray: ...

    @property
    def dim(self) -> int: ...


class SentenceTransformerEmbedding:
    """Real embedding service using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class MockEmbeddingService:
    """Fast mock for testing. Returns deterministic embeddings based on text hash."""

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
            rng = np.random.RandomState(hash(text) % (2**31))
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


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors.

    Assumes both are already normalized if using SentenceTransformerEmbedding.
    Returns 1D array of similarities.
    """
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    # Normalize just in case
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms = np.maximum(m_norms, 1e-8)
    m_normalized = matrix / m_norms
    return m_normalized @ q_norm
