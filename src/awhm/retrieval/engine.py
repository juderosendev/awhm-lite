"""RetrievalEngine: buffer check -> anchors -> neighbours -> feature ranking -> top-k.

Zero LLM calls. Lexical (BM25) and semantic (embedding) anchor sets are
unioned, one-hop graph neighbours of the anchors are pulled in with a
decayed edge weight, and every candidate is scored by a weighted blend of
semantic similarity, lexical score, strength, consolidation confidence and
neighbour evidence. While the system is cold (few sessions) a BM25 pass over
the raw logs fills the gaps.

``as_of`` turns any query into a time-travel query: only memories whose
validity window covers that moment are eligible, superseded ones included.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..config import AWHMConfig
from ..consolidation.canonical import (
    canonical_key_for_content,
    correction_supersedes,
    is_correction,
)
from ..graph.memory_graph import MemoryGraph
from ..graph.models import MemoryNode
from ..graph.strength import StrengthScorer
from ..raw_log.reader import RawLogReader
from ..session_buffer.buffer import SessionBuffer
from ..session_buffer.models import BufferEntry
from ..timeutil import parse_timestamp, utcnow, valid_at
from ..types import NodeStatus
from .bm25 import BM25Index
from .embedding import EmbeddingService, cosine_similarity_matrix
from .ranking import RankingFeatures, blended_rank_score

BUFFER_BASE_SCORE = 2.0  # Buffer hits always outrank graph results (max blend is 1.0)


@dataclass
class RetrievalResult:
    node_id: str
    content: str
    score: float
    source: str  # "buffer" | "graph" | "raw_log"
    trace: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "node_id": self.node_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
        }
        if self.trace is not None:
            data["trace"] = self.trace
        return data


class RetrievalEngine:
    """Full retrieval pipeline."""

    def __init__(
        self,
        config: AWHMConfig,
        graph: MemoryGraph,
        buffer: SessionBuffer,
        embedding_service: EmbeddingService,
        log_reader: RawLogReader,
    ) -> None:
        self.config = config
        self.graph = graph
        self.buffer = buffer
        self.embedding = embedding_service
        self.log_reader = log_reader
        self.scorer = StrengthScorer(config)
        self._bm25: BM25Index | None = None
        self._bm25_version: int | None = None
        self._bm25_node_ids: list[str] = []

    # ── Public API ─────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        k: int | None = None,
        include_history: bool | None = None,
        with_trace: bool | None = None,
        as_of: str | datetime | None = None,
        semantic: bool = True,
    ) -> list[RetrievalResult]:
        """Run the full retrieval pipeline and return the top ``k`` results.

        Args:
            k: result count (default ``config.k``).
            include_history: include superseded/retracted memories.
            with_trace: attach per-result feature traces.
            as_of: answer as of this moment; memories are filtered by their
                validity window instead of their current status.
            semantic: set False to skip embeddings entirely (BM25 and buffer
                only). Useful where loading the embedding model is too slow,
                e.g. in per-turn hooks.
        """
        k = self.config.k if k is None else k
        if include_history is None:
            include_history = self.config.include_history_by_default
        if with_trace is None:
            with_trace = self.config.trace_retrieval
        moment = parse_timestamp(as_of) if as_of is not None else None

        results = self._buffer_results(query_text, include_history, moment)
        if self.graph.node_count() > 0:
            results.extend(self._graph_retrieval(
                query_text, k, include_history, with_trace, moment, semantic,
            ))
        if self.log_reader.session_count() < self.config.cold_start_session_count:
            results.extend(self._raw_log_fallback(query_text, k, moment))

        results.sort(key=lambda r: r.score, reverse=True)
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for r in results:
            key = r.content[:200]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:k]

    # ── Step 0: session buffer ─────────────────────────────────

    def _buffer_results(
        self,
        query_text: str,
        include_history: bool,
        moment: datetime | None,
    ) -> list[RetrievalResult]:
        hits = self.buffer.search(query_text)
        if moment is not None:
            hits = [h for h in hits if parse_timestamp(h.timestamp) <= moment]
        if not include_history:
            hits = self._current_buffer_hits(hits)
        hits.sort(key=lambda e: e.source_msg, reverse=True)
        return [
            RetrievalResult(
                node_id="",
                content=hit.content,
                score=BUFFER_BASE_SCORE + hit.source_msg * 1e-4,  # newer first on ties
                source="buffer",
            )
            for hit in hits
        ]

    def _current_buffer_hits(self, hits: list[BufferEntry]) -> list[BufferEntry]:
        """Drop buffer entries that a later entry in the same session supersedes."""
        ordered = sorted(hits, key=lambda e: e.source_msg)
        keys = [canonical_key_for_content(e.content) for e in ordered]
        corrections = [is_correction(e.content) for e in ordered]
        current: list[BufferEntry] = []
        for i, entry in enumerate(ordered):
            superseded = any(
                correction_supersedes(
                    keys[j],
                    corrections[j],
                    keys[i],
                    ordered[j].source_msg - entry.source_msg,
                    self.config.correction_window_messages,
                )
                for j in range(i + 1, len(ordered))
            )
            if not superseded:
                current.append(entry)
        return current

    # ── Steps 1-3: anchors, neighbours, ranking ────────────────

    def _bm25_index(self) -> tuple[BM25Index, list[str]]:
        """BM25 over node contents, rebuilt only when nodes change."""
        if self._bm25 is None or self._bm25_version != self.graph.content_version:
            self._bm25_node_ids = self.graph.node_ids()
            self._bm25 = BM25Index(self.graph.all_contents())
            self._bm25_version = self.graph.content_version
        return self._bm25, self._bm25_node_ids

    def _eligible(self, node: MemoryNode, include_history: bool, moment: datetime | None) -> bool:
        if moment is not None:
            return valid_at(node.valid_from or node.created_at, node.valid_to, moment)
        status = node.status or NodeStatus.ACTIVE.value
        return include_history or status == NodeStatus.ACTIVE.value

    def _graph_retrieval(
        self,
        query_text: str,
        k: int,
        include_history: bool,
        with_trace: bool,
        moment: datetime | None,
        semantic: bool,
    ) -> list[RetrievalResult]:
        bm25, node_ids = self._bm25_index()
        if not node_ids:
            return []

        # Lexical anchors
        bm25_hits = bm25.search(query_text, top_k=k * 5)
        max_bm25 = max((score for _, score in bm25_hits), default=0.0)
        lexical: dict[str, float] = {}
        anchors: set[str] = set()
        for idx, score in bm25_hits:
            nid = node_ids[idx]
            lexical[nid] = score / max_bm25 if max_bm25 > 0 else 0.0
            if lexical[nid] >= self.config.bm25_anchor_ratio:
                anchors.add(nid)

        # Semantic anchors
        query_emb: np.ndarray | None = None
        semantic_sims: dict[str, float] = {}
        if semantic:
            query_emb = self.embedding.encode_single(query_text)
            emb_matrix, emb_ids = self.graph.get_embedding_matrix()
            if emb_matrix.size > 0:
                sims = cosine_similarity_matrix(query_emb, emb_matrix)
                for i, nid in enumerate(emb_ids):
                    semantic_sims[nid] = float(sims[i])
                    if semantic_sims[nid] >= self.config.embed_threshold:
                        anchors.add(nid)

        if not anchors:
            return []

        # One-hop neighbours of the anchors, carrying the strongest edge weight
        association: dict[str, float] = {}
        if self.config.neighbor_expansion:
            for nid in list(anchors):
                for edge in self.graph.get_edges_for_node(nid):
                    other = edge.target if edge.source == nid else edge.source
                    if other in anchors or other == nid:
                        continue
                    weight = max(edge.weight, 0.0) * self.config.neighbor_decay
                    if weight > association.get(other, 0.0):
                        association[other] = weight

        candidates: list[MemoryNode] = []
        for nid in list(anchors) + list(association):
            node = self.graph.get_node(nid)
            if node is not None and self._eligible(node, include_history, moment):
                candidates.append(node)
        if not candidates:
            return []

        # Strength is only needed for the candidates we are about to rank.
        now = utcnow()
        self.scorer.update_nodes(self.graph, candidates, now)

        results: list[RetrievalResult] = []
        for node in candidates:
            sim = semantic_sims.get(node.id)
            if sim is None and query_emb is not None and node.embedding:
                node_emb = np.asarray(node.embedding, dtype=np.float32).reshape(1, -1)
                sim = float(cosine_similarity_matrix(query_emb, node_emb)[0])
            status = node.status or NodeStatus.ACTIVE.value
            features = RankingFeatures(
                semantic_sim=sim or 0.0,
                lexical_score=lexical.get(node.id, 0.0),
                strength=node.strength.composite,
                confidence=min(max(float(node.confidence), 0.0), 1.0),
                status_bonus=1.0 if (moment is not None or status == NodeStatus.ACTIVE.value) else 0.0,
                association=association.get(node.id, 0.0),
            )
            trace = None
            if with_trace:
                trace = {
                    "semantic_sim": features.semantic_sim,
                    "lexical_score": features.lexical_score,
                    "strength": features.strength,
                    "confidence": features.confidence,
                    "status_bonus": features.status_bonus,
                    "association": features.association,
                }
            results.append(RetrievalResult(
                node_id=node.id,
                content=node.content,
                score=blended_rank_score(features, self.config),
                source="graph",
                trace=trace,
            ))
            if status == NodeStatus.ACTIVE.value:
                self.graph.update_node_access(node.id, now)  # reinforcement

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ── Cold-start fallback ────────────────────────────────────

    def _raw_log_fallback(
        self,
        query_text: str,
        k: int,
        moment: datetime | None,
    ) -> list[RetrievalResult]:
        """BM25 over raw log messages, scaled into ``[0, raw_log_score_scale]``."""
        documents = [
            entry.content
            for entries in self.log_reader.read_all_sessions().values()
            for entry in entries
            if moment is None or parse_timestamp(entry.timestamp) <= moment
        ]
        if not documents:
            return []
        hits = BM25Index(documents).search(query_text, top_k=k)
        if not hits:
            return []
        top = hits[0][1]
        scale = self.config.raw_log_score_scale
        return [
            RetrievalResult(
                node_id="",
                content=documents[idx],
                score=(score / top) * scale if top > 0 else 0.0,
                source="raw_log",
            )
            for idx, score in hits
        ]
