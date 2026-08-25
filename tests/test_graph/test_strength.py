"""Tests for StrengthScorer."""

from datetime import datetime, timedelta, timezone

from awhm.config import AWHMConfig
from awhm.graph.models import MemoryNode
from awhm.graph.strength import StrengthScorer


def test_recency_at_zero():
    """Just accessed -> recency = 1.0."""
    config = AWHMConfig()
    scorer = StrengthScorer(config)
    now = datetime.now(timezone.utc)
    node = MemoryNode(id="n1", content="test", last_accessed=now.isoformat())
    score = scorer.recency_score(node, now)
    assert abs(score - 1.0) < 0.01


def test_recency_decays():
    """Score should decay over time."""
    config = AWHMConfig()
    scorer = StrengthScorer(config)
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=24)
    node = MemoryNode(id="n1", content="test", last_accessed=old.isoformat())
    score = scorer.recency_score(node, now)
    assert score < 1.0
    assert score > 0.0


def test_recency_24h():
    """At 24h: ~0.71 per spec."""
    config = AWHMConfig()
    scorer = StrengthScorer(config)
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(hours=24)
    node = MemoryNode(id="n1", content="test", last_accessed=day_ago.isoformat())
    score = scorer.recency_score(node, now)
    assert 0.60 < score < 0.80  # ~0.71


def test_frequency_score():
    config = AWHMConfig()
    scorer = StrengthScorer(config)
    node = MemoryNode(id="n1", content="test", access_count=5)
    score = scorer.frequency_score(node, p90_access=10.0)
    assert abs(score - 0.5) < 0.01


def test_frequency_cap():
    config = AWHMConfig()
    scorer = StrengthScorer(config)
    node = MemoryNode(id="n1", content="test", access_count=20)
    score = scorer.frequency_score(node, p90_access=10.0)
    assert score == 1.0


def test_composite(sample_graph):
    config = AWHMConfig()
    scorer = StrengthScorer(config)
    scorer.update_all(sample_graph)
    for node in sample_graph.nodes.values():
        assert 0 <= node.strength.composite <= 1.5  # Reasonable range
