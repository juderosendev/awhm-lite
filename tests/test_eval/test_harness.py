"""Tests for benchmark harness."""

from awhm.eval.harness import run_builtin_benchmark


def test_builtin_benchmark_runs():
    report = run_builtin_benchmark(k=5, use_mock_embeddings=True)
    assert report["query_count"] >= 3
    assert 0.0 <= report["recall_at_k"] <= 1.0
    assert 0.0 <= report["ndcg_at_k"] <= 1.0
    assert 0.0 <= report["contradiction_error_rate"] <= 1.0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"]
    assert "deletion_audit" in report
    assert report["deletion_audit"]["passed"] is True
