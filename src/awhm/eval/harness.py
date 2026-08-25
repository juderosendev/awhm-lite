"""Replay-style benchmark harness for memory quality metrics."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any

from .. import AWHMSession
from ..config import AWHMConfig
from ..raw_log.reader import RawLogReader
from ..types import Role


@dataclass
class EvalQueryCase:
    name: str
    query: str
    expected_substring: str
    forbidden_substring: str | None = None


def run_builtin_benchmark(
    *,
    k: int = 5,
    use_mock_embeddings: bool = True,
) -> dict[str, Any]:
    """Run a deterministic benchmark over synthetic correction-heavy sessions."""
    with TemporaryDirectory(prefix="awhm_eval_") as tmpdir:
        config = AWHMConfig(
            data_dir=tmpdir,
            cold_start_session_count=0,
            k=max(k, 1),
        )

        _seed_benchmark_data(config, use_mock_embeddings=use_mock_embeddings)
        report = _run_queries_and_metrics(
            config,
            use_mock_embeddings=use_mock_embeddings,
            k=max(k, 1),
        )
        report["deletion_audit"] = _run_deletion_audit(
            config,
            use_mock_embeddings=use_mock_embeddings,
            k=max(k, 1),
        )
        return report


def _seed_benchmark_data(config: AWHMConfig, use_mock_embeddings: bool) -> None:
    s1 = AWHMSession.start_session(
        config,
        session_id="baseline",
        use_mock_embeddings=use_mock_embeddings,
    )
    s1.log_message(Role.USER, "My preferred language is Python")
    s1.log_message(Role.USER, "The API endpoint is https://api.v1.example.com")
    s1.log_message(Role.USER, "My city is San Francisco")
    s1.consolidate_current()
    s1.end_session()

    s2 = AWHMSession.start_session(
        config,
        session_id="corrections",
        use_mock_embeddings=use_mock_embeddings,
    )
    s2.log_message(Role.USER, "Actually, my preferred language is Rust")
    s2.log_message(Role.USER, "Actually, the API endpoint is https://api.v2.example.com")
    s2.log_message(Role.USER, "Actually, my city is New York")
    s2.consolidate_current()
    s2.end_session()


def _run_queries_and_metrics(
    config: AWHMConfig,
    *,
    use_mock_embeddings: bool,
    k: int,
) -> dict[str, Any]:
    query_cases = [
        EvalQueryCase(
            name="preferred_language",
            query="What is my preferred language?",
            expected_substring="rust",
            forbidden_substring="python",
        ),
        EvalQueryCase(
            name="api_endpoint",
            query="What is the API endpoint?",
            expected_substring="v2",
            forbidden_substring="v1",
        ),
        EvalQueryCase(
            name="city",
            query="What is my city?",
            expected_substring="new york",
            forbidden_substring="san francisco",
        ),
    ]

    evaluator = AWHMSession.start_session(
        config,
        session_id="eval",
        use_mock_embeddings=use_mock_embeddings,
    )

    case_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    recall_hits = 0
    ndcg_sum = 0.0
    contradiction_hits = 0

    for case in query_cases:
        start = time.perf_counter()
        results = evaluator.query(case.query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        hit_rank = _first_match_rank(results, case.expected_substring)
        forbidden_hit = _contains_any(results, case.forbidden_substring)
        recall = 1.0 if hit_rank is not None else 0.0
        ndcg = 1.0 / math.log2(hit_rank + 1) if hit_rank is not None else 0.0

        recall_hits += int(recall)
        ndcg_sum += ndcg
        contradiction_hits += int(forbidden_hit)

        case_results.append({
            "name": case.name,
            "query": case.query,
            "latency_ms": latency_ms,
            "recall_hit": bool(recall),
            "hit_rank": hit_rank,
            "ndcg": ndcg,
            "forbidden_hit": forbidden_hit,
            "top_results": [r.to_dict() for r in results[:3]],
        })

    evaluator.end_session()

    n_cases = len(query_cases)
    return {
        "version": "builtin-v1",
        "query_count": n_cases,
        "k": k,
        "recall_at_k": recall_hits / n_cases if n_cases else 0.0,
        "ndcg_at_k": ndcg_sum / n_cases if n_cases else 0.0,
        "contradiction_error_rate": contradiction_hits / n_cases if n_cases else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
        },
        "cases": case_results,
    }


def _run_deletion_audit(
    config: AWHMConfig,
    *,
    use_mock_embeddings: bool,
    k: int,
) -> dict[str, Any]:
    session = AWHMSession.start_session(
        config,
        session_id="privacy",
        use_mock_embeddings=use_mock_embeddings,
    )
    secret = "alpha-123-secret"
    session.log_message(Role.USER, f"My token is {secret}")
    session.consolidate_current()
    session.create_snapshot()

    target_id = None
    for node in session.graph.all_nodes():
        if secret in node.content.lower():
            target_id = node.id
            break

    if target_id is None:
        session.end_session()
        return {
            "passed": False,
            "reason": "secret_node_not_found",
        }

    deletion = session.delete_node(target_id)
    post_query = session.query(secret, k=k, include_history=True)
    session.end_session()

    reader = RawLogReader(config)
    logs_clean = True
    for entries in reader.read_all_sessions().values():
        for entry in entries:
            if secret in entry.content.lower():
                logs_clean = False
                break
        if not logs_clean:
            break

    snapshots_clean = _snapshots_absent(config, secret)
    query_clean = not any(secret in r.content.lower() for r in post_query)
    passed = logs_clean and snapshots_clean and query_clean

    return {
        "passed": passed,
        "logs_clean": logs_clean,
        "snapshots_clean": snapshots_clean,
        "query_clean": query_clean,
        "deletion_result": {
            "node_deleted": deletion.node_deleted,
            "snapshots_touched": deletion.snapshots_touched,
            "tombstone_id": deletion.tombstone_id,
            "ledger_record_id": deletion.ledger_record_id,
        },
    }


def _snapshots_absent(config: AWHMConfig, token: str) -> bool:
    token = token.lower()
    for path in config.snapshots_dir.glob("snapshot_*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        graph = data.get("graph", {})
        nodes = graph.get("nodes", {})
        if isinstance(nodes, dict):
            for node in nodes.values():
                if not isinstance(node, dict):
                    continue
                if token in str(node.get("content", "")).lower():
                    return False
        wal_state = data.get("wal_state")
        if isinstance(wal_state, list):
            for entry in wal_state:
                if token in str(entry.get("content", "")).lower():
                    return False
    return True


def _contains_any(results: list[Any], substring: str | None) -> bool:
    if not substring:
        return False
    target = substring.lower()
    return any(target in str(r.content).lower() for r in results)


def _first_match_rank(results: list[Any], expected_substring: str) -> int | None:
    target = expected_substring.lower()
    for i, result in enumerate(results, start=1):
        if target in str(result.content).lower():
            return i
    return None




def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight
