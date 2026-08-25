"""StrengthScorer: recency (power-law decay), frequency, and composite scores."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from ..config import AWHMConfig
from ..timeutil import parse_timestamp
from .memory_graph import MemoryGraph
from .models import MemoryNode


class StrengthScorer:
    """Compute S(v) = w_rec * s_rec(v) + w_freq * s_freq(v)."""

    def __init__(self, config: AWHMConfig) -> None:
        self.config = config

    def recency_score(self, node: MemoryNode, now: datetime | None = None) -> float:
        """s_rec(v) = (1 + beta * delta_t) ** (-alpha), with delta_t in hours."""
        now = now or datetime.now(timezone.utc)
        last = parse_timestamp(node.last_accessed)
        delta_hours = max((now - last).total_seconds() / 3600.0, 0.0)
        return (1.0 + self.config.beta * delta_hours) ** (-self.config.alpha)

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
        """90th percentile of access counts across the graph (at least 1)."""
        if not graph.nodes:
            return 1.0
        counts = sorted(n.access_count for n in graph.nodes.values())
        idx = min(int(len(counts) * 0.9), len(counts) - 1)
        return max(float(counts[idx]), 1.0)

    def update_nodes(
        self,
        graph: MemoryGraph,
        nodes: Iterable[MemoryNode],
        now: datetime | None = None,
    ) -> None:
        """Recompute and store strength for a subset of nodes."""
        now = now or datetime.now(timezone.utc)
        p90 = self.compute_p90_access(graph)
        for node in nodes:
            s_rec = self.recency_score(node, now)
            s_freq = self.frequency_score(node, p90)
            node.strength.recency = s_rec
            node.strength.frequency = node.access_count
            node.strength.composite = self.config.w_rec * s_rec + self.config.w_freq * s_freq

    def update_all(self, graph: MemoryGraph, now: datetime | None = None) -> None:
        """Recompute strength scores for every node in the graph."""
        self.update_nodes(graph, graph.nodes.values(), now)
