"""Tests for ranking formula."""

from awhm.config import AWHMConfig
from awhm.retrieval.ranking import rank_score


def test_rank_score_high_similarity():
    config = AWHMConfig()
    score = rank_score(similarity=0.9, strength=0.8, config=config)
    assert score > 0


def test_rank_score_zero_similarity():
    config = AWHMConfig()
    score = rank_score(similarity=0.0, strength=1.0, config=config)
    assert score == 0.0


def test_rank_score_increases_with_similarity():
    config = AWHMConfig()
    low = rank_score(similarity=0.3, strength=0.5, config=config)
    high = rank_score(similarity=0.9, strength=0.5, config=config)
    assert high > low


def test_rank_score_strength_matters():
    config = AWHMConfig()
    weak = rank_score(similarity=0.7, strength=0.1, config=config)
    strong = rank_score(similarity=0.7, strength=0.9, config=config)
    assert strong > weak
