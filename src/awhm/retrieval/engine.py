"""RetrievalEngine: buffer check -> anchors -> rank -> top-k + cold-start fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import AWHMConfig
from ..consolidation.canonical import canonical_key_for_content
from ..graph.memory_graph import MemoryGraph
from ..graph.strength import StrengthScorer
from ..raw_log.reader import RawLogReader
from ..session_buffer.buffer import SessionBuffer
from ..session_buffer.models import BufferEntry
from ..types import NodeStatus
from .bm25 import BM25Index
from .embedding import EmbeddingService, cosine_similarity_matrix
from .ranking import RankingFeatures, blended_rank_score


@dataclass
class RetrievalResult:
    node_id: str
    content: str
    score: float
    source: str  # "buffer", "graph", "raw_log"
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
    """Full retrieval pipeline per spec section 6."""

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

    def query(
        self,
        query_text: str,
        k: int | None = None,
        include_history: bool | None = None,
        with_trace: bool | None = None,
    ) -> list[RetrievalResult]:
        """Full retrieval pipeline.

        Step 0: Buffer check
        Step 1: Anchor identification (BM25 + embedding)
        Step 2: Rank by R(v,q)
        Cold-start fallback: BM25 over raw logs if <10 sessions
        """
        if k is None:
            k = self.config.k
        if include_history is None:
            include_history = self.config.include_history_by_default
        if with_trace is None:
            with_trace = self.config.trace_retrieval
        results: list[RetrievalResult] = []

        # Step 0: Session buffer check
        buffer_hits = self.buffer.search(query_text)
        if not include_history:
            buffer_hits = self._latest_buffer_hits(buffer_hits)
        buffer_hits.sort(key=lambda e: e.source_msg, reverse=True)
        for bh in buffer_hits:
            results.append(RetrievalResult(
                node_id="",
                content=bh.content,
                score=2.0 + (bh.source_msg * 1e-4),  # Buffer hits get priority + recency tie-break
                source="buffer",
            ))

        # Step 1+2: Graph retrieval
        if self.graph.node_count() > 0:
            graph_results = self._graph_retrieval(
                query_text,
                k,
                include_history=include_history,
                with_trace=with_trace,
            )
            results.extend(graph_results)

        # Cold-start fallback: BM25 over raw logs
        session_count = self.log_reader.session_count()
        if session_count < self.config.cold_start_session_count:
            log_results = self._raw_log_fallback(query_text, k)
            results.extend(log_results)

        # Sort by score descending, deduplicate by content
        results.sort(key=lambda r: r.score, reverse=True)
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for r in results:
            key = r.content[:200]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:k]

    def _graph_retrieval(
        self,
        query_text: str,
        k: int,
        include_history: bool = False,
        with_trace: bool = False,
    ) -> list[RetrievalResult]:
        """BM25 + embedding anchor identification, then rank."""
        # Refresh strength scores
        self.scorer.update_all(self.graph)

        node_ids = self.graph.node_ids()
        contents = self.graph.all_contents()
        if not contents:
            return []

        # BM25 anchors
        bm25_index = BM25Index(contents)
        bm25_hits = bm25_index.search(query_text, top_k=k * 5)
        bm25_anchors: set[int] = set()
        lexical_scores: dict[str, float] = {}
        max_bm25 = max((score for _, score in bm25_hits), default=0.0)
        for idx, score in bm25_hits:
            if idx < len(node_ids) and max_bm25 > 0:
                lexical_scores[node_ids[idx]] = float(score / max_bm25)
            if score >= self.config.bm25_threshold:
                bm25_anchors.add(idx)

        # Embedding anchors
        query_emb = self.embedding.encode_single(query_text)
        emb_matrix, emb_ids = self.graph.get_embedding_matrix()

        embed_anchors: set[str] = set()
        sim_map: dict[str, float] = {}

        if emb_matrix.size > 0:
            sims = cosine_similarity_matrix(query_emb, emb_matrix)
            for i, nid in enumerate(emb_ids):
                sim_val = float(sims[i])
                sim_map[nid] = sim_val
                if sim_val >= self.config.embed_threshold:
                    embed_anchors.add(nid)

        # Union of anchor sets
        anchor_node_ids: set[str] = set()
        for idx in bm25_anchors:
            if idx < len(node_ids):
                anchor_node_ids.add(node_ids[idx])
        anchor_node_ids |= embed_anchors

        if not anchor_node_ids:
            return []

        # Step 2: Rank
        results: list[RetrievalResult] = []
        for nid in anchor_node_ids:
            node = self.graph.get_node(nid)
            if node is None:
                continue
            node_status = node.status or NodeStatus.ACTIVE.value
            if not include_history and node_status != NodeStatus.ACTIVE.value:
                continue

            sim = sim_map.get(nid, 0.0)
            if sim <= 0 and nid not in embed_anchors:
                # BM25-only anchor: compute embedding similarity
                if node.embedding:
                    node_emb = np.array(node.embedding, dtype=np.float32)
                    sim = float(cosine_similarity_matrix(query_emb, node_emb.reshape(1, -1))[0])
                    sim_map[nid] = sim
            status_bonus = 1.0 if node_status == NodeStatus.ACTIVE.value else 0.0
            features = RankingFeatures(
                semantic_sim=sim,
                lexical_score=lexical_scores.get(nid, 0.0),
                strength=node.strength.composite,
                confidence=min(max(float(node.confidence), 0.0), 1.0),
                status_bonus=status_bonus,
            )
            score = blended_rank_score(features, self.config)
            trace = None
            if with_trace:
                trace = {
                    "semantic_sim": features.semantic_sim,
                    "lexical_score": features.lexical_score,
                    "strength": features.strength,
                    "confidence": features.confidence,
                    "status_bonus": features.status_bonus,
                }
            results.append(RetrievalResult(
                node_id=nid,
                content=node.content,
                score=score,
                source="graph",
                trace=trace,
            ))
            # Reinforcement: update access on retrieval hit
            if node_status == NodeStatus.ACTIVE.value:
                self.graph.update_node_access(nid)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _raw_log_fallback(self, query_text: str, k: int) -> list[RetrievalResult]:
        """BM25 over raw log contents for cold-start."""
        all_sessions = self.log_reader.read_all_sessions()
        documents: list[str] = []
        for entries in all_sessions.values():
            for entry in entries:
                documents.append(entry.content)

        if not documents:
            return []

        bm25 = BM25Index(documents)
        hits = bm25.search(query_text, top_k=k)

        results: list[RetrievalResult] = []
        for idx, score in hits:
            results.append(RetrievalResult(
                node_id="",
                content=documents[idx],
                score=score * 0.5,  # Discount raw log results
                source="raw_log",
            ))
        return results

    def _latest_buffer_hits(self, hits: list[BufferEntry]) -> list[BufferEntry]:
        """Collapse contradictory buffer hits by canonical key, keeping latest."""
        latest_by_key: dict[str, BufferEntry] = {}
        passthrough: list[BufferEntry] = []
        for hit in hits:
            memory_type = self._buffer_entry_memory_type(hit.type)
            key = canonical_key_for_content(hit.content, memory_type)
            if not key:
                passthrough.append(hit)
                continue
            existing = latest_by_key.get(key)
            if existing is None or hit.source_msg >= existing.source_msg:
                latest_by_key[key] = hit
        return list(latest_by_key.values()) + passthrough

    @staticmethod
    def _buffer_entry_memory_type(entry_type: str) -> str:
        if entry_type in ("fact", "correction"):
            return "semantic"
        if entry_type == "preference":
            return "procedural"
        return "episodic"
