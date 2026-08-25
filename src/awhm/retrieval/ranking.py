"""Retrieval feature scoring and backward-compatible ranking helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AWHMConfig


@dataclass
class RankingFeatures:
    semantic_sim: float
    lexical_score: float
    strength: float
    confidence: float
    status_bonus: float = 1.0


def blended_rank_score(features: RankingFeatures, config: AWHMConfig) -> float:
    """Blend retrieval features into a final score.

    Score components are additive, with a contradiction penalty for
    non-active memories when history is explicitly included.
    """
    base = (
        config.w_semantic * max(features.semantic_sim, 0.0)
        + config.w_lexical * max(features.lexical_score, 0.0)
        + config.w_strength * max(features.strength, 0.0)
        + config.w_confidence * max(features.confidence, 0.0)
    )
    penalty = (1.0 - min(max(features.status_bonus, 0.0), 1.0)) * config.contradiction_penalty
    return max(base - penalty, 0.0)


def rank_score(
    similarity: float,
    strength: float,
    config: AWHMConfig,
) -> float:
    """Backward-compatible rank helper used by older tests/callers.

    Maps legacy two-signal ranking into the new feature blender.
    """
    if similarity <= 0:
        return 0.0
    return blended_rank_score(
        RankingFeatures(
            semantic_sim=similarity,
            lexical_score=0.0,
            strength=strength,
            confidence=0.6,
            status_bonus=1.0,
        ),
        config,
    )
