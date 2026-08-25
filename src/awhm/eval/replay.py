"""Replay evaluation on real corpora.

A corpus is a set of sessions to consolidate plus questions with expected
answers. The harness replays the sessions through a fresh memory system,
asks every question, and reports Recall@k, nDCG@k, contradiction rate and
latency. Two input formats:

* the native format (``sessions`` + ``questions``), see ``load_corpus``;
* LongMemEval (``haystack_sessions`` + ``question`` + ``answer``), see
  ``load_longmemeval``. Matching is by answer substring, which is a
  deliberate approximation: it under-counts paraphrased hits and is
  therefore a lower bound on recall.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..config import AWHMConfig
from ..types import Role


@dataclass
class Session:
    id: str
    messages: list[dict[str, str]]  # {"role": ..., "content": ...}


@dataclass
class Question:
    id: str
    question: str
    expected: list[str]              # any of these substrings counts as a hit
    forbidden: list[str] = field(default_factory=list)  # any of these counts as a contradiction
    as_of: str | None = None
    category: str | None = None


@dataclass
class Corpus:
    sessions: list[Session]
    questions: list[Question]
    name: str = "corpus"


# ── Loaders ────────────────────────────────────────────────────


def load_corpus(path: str | Path) -> Corpus:
    """Native format::

        {"sessions": [{"id": "s1", "messages": [{"role": "user", "content": "..."}]}],
         "questions": [{"id": "q1", "question": "...", "expected": ["..."],
                        "forbidden": ["..."], "as_of": null, "category": null}]}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sessions = [
        Session(id=str(s.get("id", i)), messages=list(s.get("messages", [])))
        for i, s in enumerate(data.get("sessions", []))
    ]
    questions = [
        Question(
            id=str(q.get("id", i)),
            question=str(q["question"]),
            expected=[str(x) for x in (q.get("expected") or [])],
            forbidden=[str(x) for x in (q.get("forbidden") or [])],
            as_of=q.get("as_of"),
            category=q.get("category"),
        )
        for i, q in enumerate(data.get("questions", []))
    ]
    return Corpus(sessions=sessions, questions=questions, name=Path(path).stem)


def load_longmemeval(path: str | Path, limit: int | None = None) -> Corpus:
    """LongMemEval JSON: a list of instances, each with ``question``, ``answer``,
    ``haystack_session_ids`` and ``haystack_sessions`` (lists of turns).

    Every instance's haystack becomes its own set of sessions, namespaced by
    the question id so instances do not bleed into each other.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or data.get("instances") or []
    sessions: list[Session] = []
    questions: list[Question] = []
    for i, inst in enumerate(data[: limit] if limit else data):
        qid = str(inst.get("question_id", i))
        ids = inst.get("haystack_session_ids") or []
        for j, turns in enumerate(inst.get("haystack_sessions") or []):
            sid = f"{qid}/{ids[j] if j < len(ids) else j}"
            messages = [
                {"role": t.get("role", "user"), "content": str(t.get("content", ""))}
                for t in turns
                if isinstance(t, dict)
            ]
            sessions.append(Session(id=sid, messages=messages))
        answer = inst.get("answer")
        expected = [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
        questions.append(Question(
            id=qid,
            question=str(inst.get("question", "")),
            expected=[a for a in expected if a],
            category=inst.get("question_type"),
        ))
    return Corpus(sessions=sessions, questions=questions, name=Path(path).stem)


# ── Runner ─────────────────────────────────────────────────────


def _role(value: str) -> Role:
    try:
        return Role(value)
    except ValueError:
        return Role.ASSISTANT if value.lower().startswith("assist") else Role.USER


def _first_hit(contents: list[str], needles: list[str]) -> int | None:
    lowered = [c.lower() for c in contents]
    for rank, text in enumerate(lowered, start=1):
        if any(n.lower() in text for n in needles):
            return rank
    return None


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    rank = (len(values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (rank - low)


def run_replay(
    corpus: Corpus,
    *,
    k: int = 5,
    config: AWHMConfig | None = None,
    use_mock_embeddings: bool = False,
    llm_client: Any | None = None,
    session_scope: bool = True,
) -> dict[str, Any]:
    """Replay ``corpus`` through fresh memory systems and score every question.

    With ``session_scope`` (default), sessions whose ids carry an instance
    prefix (``"<question_id>/<session>"``, as produced by ``load_longmemeval``)
    are consolidated into an isolated store per instance, and each question
    is asked only against its own instance. Sessions without a prefix share
    one store and see every question. Set ``session_scope=False`` to put
    everything in one store regardless.
    """
    from .. import AWHMSession

    base = config or AWHMConfig()

    groups: dict[str, list[Session]] = {}
    for session in corpus.sessions:
        prefix = session.id.split("/", 1)[0] if session_scope and "/" in session.id else ""
        groups.setdefault(prefix, []).append(session)
    questions_by_group: dict[str, list[Question]] = {}
    for q in corpus.questions:
        target = q.id if q.id in groups else ""
        questions_by_group.setdefault(target, []).append(q)

    cases: list[dict[str, Any]] = []
    latencies: list[float] = []
    hits = ndcg_sum = contradictions = 0.0
    by_category: dict[str, dict[str, float]] = {}
    total_nodes = 0
    ingest_seconds = 0.0

    with TemporaryDirectory(prefix="awhm_replay_") as tmpdir:
        for prefix, sessions in groups.items():
            store = Path(tmpdir) / (prefix or "shared")
            cfg = AWHMConfig(**{
                **base.__dict__, "data_dir": str(store), "cold_start_session_count": 0,
            })
            start = time.perf_counter()
            nodes = 0
            for session in sessions:
                with AWHMSession.start_session(
                    cfg, session_id=session.id, use_mock_embeddings=use_mock_embeddings,
                    llm_client=llm_client,
                ) as s:
                    for m in session.messages:
                        s.log_message(_role(str(m.get("role", "user"))), str(m.get("content", "")))
                    s.consolidate_current()
                    nodes = s.graph.node_count()
            ingest_seconds += time.perf_counter() - start
            total_nodes += nodes

            questions = questions_by_group.get(prefix, [])
            if not questions:
                continue
            evaluator = AWHMSession.start_session(
                cfg, session_id="__eval__", use_mock_embeddings=use_mock_embeddings,
            )
            try:
                for q in questions:
                    start = time.perf_counter()
                    results = evaluator.query(q.question, k=k, as_of=q.as_of)
                    latency = (time.perf_counter() - start) * 1000.0
                    latencies.append(latency)
                    contents = [r.content for r in results]
                    rank = _first_hit(contents, q.expected) if q.expected else None
                    forbidden_hit = bool(q.forbidden) and _first_hit(contents, q.forbidden) is not None
                    ndcg = 1.0 / math.log2(rank + 1) if rank else 0.0
                    hits += 1.0 if rank else 0.0
                    ndcg_sum += ndcg
                    contradictions += 1.0 if forbidden_hit else 0.0
                    cat = by_category.setdefault(q.category or "all", {"n": 0.0, "hits": 0.0})
                    cat["n"] += 1
                    cat["hits"] += 1.0 if rank else 0.0
                    cases.append({
                        "id": q.id, "question": q.question, "category": q.category,
                        "hit_rank": rank, "ndcg": ndcg, "forbidden_hit": forbidden_hit,
                        "latency_ms": latency, "top_results": contents[:3],
                    })
            finally:
                evaluator.end_session()

    n = len(corpus.questions)
    return {
        "corpus": corpus.name,
        "sessions": len(corpus.sessions),
        "messages": sum(len(s.messages) for s in corpus.sessions),
        "questions": n,
        "k": k,
        "nodes": total_nodes,
        "ingest_seconds": ingest_seconds,
        "recall_at_k": hits / n if n else 0.0,
        "ndcg_at_k": ndcg_sum / n if n else 0.0,
        "contradiction_error_rate": contradictions / n if n else 0.0,
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "by_category": {
            name: {"questions": int(v["n"]), "recall_at_k": v["hits"] / v["n"] if v["n"] else 0.0}
            for name, v in sorted(by_category.items())
        },
        "cases": cases,
    }


def summarize(report: dict[str, Any]) -> str:
    lines = [
        f"AWHM replay: {report['corpus']}",
        f"  sessions {report['sessions']}, messages {report['messages']}, nodes {report['nodes']}, "
        f"ingest {report['ingest_seconds']:.1f}s",
        f"  Recall@{report['k']}: {report['recall_at_k']:.3f}",
        f"  nDCG@{report['k']}: {report['ndcg_at_k']:.3f}",
        f"  Contradiction error rate: {report['contradiction_error_rate']:.3f}",
        f"  Latency p50/p95 (ms): {report['latency_ms']['p50']:.1f} / {report['latency_ms']['p95']:.1f}",
    ]
    for name, v in report.get("by_category", {}).items():
        if name != "all":
            lines.append(f"    {name}: recall {v['recall_at_k']:.3f} over {v['questions']}")
    return "\n".join(lines)
