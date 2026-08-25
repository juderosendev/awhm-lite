"""Tests for replay evaluation and corpus loaders."""

import json

from awhm.eval.replay import load_corpus, load_longmemeval, run_replay, summarize


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_native_corpus_replay(tmp_path):
    path = _write(tmp_path, "corpus.json", {
        "sessions": [
            {"id": "s1", "messages": [
                {"role": "user", "content": "My preferred language is Python"},
                {"role": "assistant", "content": "Noted."},
            ]},
            {"id": "s2", "messages": [
                {"role": "user", "content": "Actually, my preferred language is Rust"},
            ]},
        ],
        "questions": [
            {"id": "q1", "question": "What is my preferred language?",
             "expected": ["rust"], "forbidden": ["python"], "category": "update"},
            {"id": "q2", "question": "What colour is the sky?", "expected": ["blue"]},
        ],
    })
    corpus = load_corpus(path)
    assert len(corpus.sessions) == 2 and len(corpus.questions) == 2

    report = run_replay(corpus, k=5, use_mock_embeddings=True, session_scope=False)
    assert report["questions"] == 2
    assert report["recall_at_k"] == 0.5
    assert report["contradiction_error_rate"] == 0.0
    assert report["by_category"]["update"]["recall_at_k"] == 1.0
    assert report["cases"][0]["hit_rank"] == 1
    assert "Recall@5: 0.500" in summarize(report)


def test_longmemeval_loader_and_scoping(tmp_path):
    path = _write(tmp_path, "lme.json", [
        {
            "question_id": "a",
            "question_type": "single-session-user",
            "question": "What is my preferred language?",
            "answer": "Rust",
            "haystack_session_ids": ["h1"],
            "haystack_sessions": [[
                {"role": "user", "content": "My preferred language is Rust"},
                {"role": "assistant", "content": "Great choice."},
            ]],
        },
        {
            "question_id": "b",
            "question_type": "single-session-user",
            "question": "What is my preferred language?",
            "answer": ["Go"],
            "haystack_session_ids": ["h1"],
            "haystack_sessions": [[{"role": "user", "content": "My preferred language is Go"}]],
        },
    ])
    corpus = load_longmemeval(path)
    assert [s.id for s in corpus.sessions] == ["a/h1", "b/h1"]
    assert corpus.questions[1].expected == ["Go"]

    # Each instance is isolated in its own store, so the two instances'
    # conflicting "preferred language" facts do not supersede each other.
    report = run_replay(corpus, k=5, use_mock_embeddings=True)
    assert report["recall_at_k"] == 1.0
    # Without isolation the later instance supersedes the earlier one.
    assert run_replay(corpus, k=5, use_mock_embeddings=True, session_scope=False)["recall_at_k"] == 0.5
    assert report["by_category"]["single-session-user"]["questions"] == 2

    assert len(load_longmemeval(path, limit=1).questions) == 1
