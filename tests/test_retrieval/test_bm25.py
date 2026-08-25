"""Tests for BM25 index."""

from awhm.retrieval.bm25 import BM25Index, tokenize


def test_tokenize():
    tokens = tokenize("The quick brown FOX jumped over the lazy dog")
    assert "quick" in tokens
    assert "fox" in tokens
    assert "the" not in tokens  # stopword


def test_bm25_search():
    docs = [
        "Python programming language",
        "JavaScript web development",
        "Python data science numpy",
    ]
    index = BM25Index(docs)
    results = index.search("Python programming")
    assert len(results) > 0
    # First result should be the Python programming doc
    assert results[0][0] == 0


def test_bm25_empty():
    index = BM25Index()
    results = index.search("test")
    assert results == []


def test_bm25_document_count():
    index = BM25Index(["doc1", "doc2", "doc3"])
    assert index.document_count == 3
