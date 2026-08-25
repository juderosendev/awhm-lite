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


def test_bm25_only_returns_documents_containing_query_terms():
    # "python" appears in every document, so classic IDF would go negative.
    docs = ["python one", "python two", "python three"]
    index = BM25Index(docs)
    results = index.search("python two")
    assert results and results[0][0] == 1
    assert all(score > 0 for _, score in results)
    assert index.search("golang") == []


def test_bm25_scores_are_positive_for_shared_terms():
    index = BM25Index(["alpha beta", "alpha gamma"])
    scores = index.scores("alpha")
    assert all(s > 0 for s in scores)
