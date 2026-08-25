"""StrengthScorer: recency (power-law), frequency, composite scoring."""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import AWHMConfig
from .memory_graph import MemoryGraph
from .models import MemoryNode


class StrengthScorer:
    """Compute S(v) = w_rec * s_rec(v) + w_freq * s_freq(v)."""

    def __init__(self, config: AWHMConfig) -> None:
        self.config = config

    def recency_score(self, node: MemoryNode, now: datetime | None = None) -> float:
        """s_rec(v) = (1 + beta * delta_t)^(-alpha)

        delta_t in hours.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        last = datetime.fromisoformat(node.last_accessed)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta_hours = max((now - last).total_seconds() / 3600, 0)
        return (1 + self.config.beta * delta_hours) ** (-self.config.alpha)

    def frequency_score(self, node: MemoryNode, p90_access: float) -> float:
        """s_freq(v) = min(access_count / p90_access_count, 1.0)."""
        if p90_access <= 0:
            return 1.0
        return min(node.access_count / p90_access, 1.0)

    def composite_score(
        self, node: MemoryNode, p90_access: float, now: datetime | None = None
    ) -> float:
        """S(v) = w_rec * s_rec + w_freq * s_freq."""
        s_rec = self.recency_score(node, now)
        s_freq = self.frequency_score(node, p90_access)
        return self.config.w_rec * s_rec + self.config.w_freq * s_freq

    def compute_p90_access(self, graph: MemoryGraph) -> float:
        """Compute 90th percentile access count across all nodes."""
        if not graph.nodes:
            return 1.0
        counts = sorted(n.access_count for n in graph.nodes.values())
        idx = int(len(counts) * 0.9)
        idx = min(idx, len(counts) - 1)
        return max(counts[idx], 1.0)

    def update_all(self, graph: MemoryGraph, now: datetime | None = None) -> None:
        """Recompute strength scores for all nodes in the graph."""
        p90 = self.compute_p90_access(graph)
        for node in graph.nodes.values():
            s_rec = self.recency_score(node, now)
            s_freq = self.frequency_score(node, p90)
            node.strength.recency = s_rec
            node.strength.frequency = node.access_count
            node.strength.composite = (
                self.config.w_rec * s_rec + self.config.w_freq * s_freq
            )
