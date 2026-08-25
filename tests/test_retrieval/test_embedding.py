"""Tests for embedding service."""

import numpy as np

from awhm.retrieval.embedding import (
    MockEmbeddingService,
    cosine_similarity,
    cosine_similarity_matrix,
)


def test_mock_encode():
    svc = MockEmbeddingService(dim=384)
    result = svc.encode(["hello", "world"])
    assert result.shape == (2, 384)


def test_mock_encode_single():
    svc = MockEmbeddingService(dim=384)
    result = svc.encode_single("hello")
    assert result.shape == (384,)


def test_mock_deterministic():
    svc = MockEmbeddingService(dim=384)
    a = svc.encode_single("hello")
    b = svc.encode_single("hello")
    assert np.allclose(a, b)


def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_matrix():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = cosine_similarity_matrix(query, matrix)
    assert abs(sims[0] - 1.0) < 1e-5
    assert abs(sims[1]) < 1e-5
