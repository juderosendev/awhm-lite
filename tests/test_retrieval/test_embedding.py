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


def test_mock_deterministic_across_processes():
    import json
    import subprocess
    import sys

    code = (
        "import json; from awhm.retrieval.embedding import MockEmbeddingService; "
        "print(json.dumps(MockEmbeddingService(dim=8).encode_single('hello').tolist()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    other = np.array(json.loads(out.stdout))
    assert np.allclose(other, MockEmbeddingService(dim=8).encode_single("hello"), atol=1e-6)


def test_sentence_transformer_model_is_shared_per_process(monkeypatch):
    import sys
    import types

    from awhm.retrieval import embedding as emb

    loads = []

    class FakeModel:
        def __init__(self, name):
            loads.append(name)

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            return np.ones((len(texts), 4), dtype=np.float32)

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(emb, "_MODEL_CACHE", {})

    a = emb.SentenceTransformerEmbedding("fake-model")
    b = emb.SentenceTransformerEmbedding("fake-model")
    assert a.encode(["x"]).shape == (1, 4)
    assert b.encode_single("y").shape == (4,)
    assert loads == ["fake-model"]
