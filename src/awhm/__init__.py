"""AWHM Lite: external long-term memory for LLM agents.

Everything runs locally with zero LLM calls. The top-level entry point is
:class:`AWHMSession`::

    from awhm import AWHMSession, Role

    with AWHMSession.start_session() as session:
        session.log_message(Role.USER, "My name is Alice")
        for result in session.query("What is the user's name?"):
            print(result.content)
        session.consolidate_current()
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from .config import AWHMConfig
from .consolidation.pipeline import Stage1Pipeline
from .deletion.hard_delete import DeletionResult, hard_delete
from .graph.memory_graph import MemoryGraph
from .graph.serialization import load_graph, save_graph
from .graph.strength import StrengthScorer
from .raw_log.logger import RawLogger
from .raw_log.reader import RawLogReader
from .retrieval.bm25 import BM25Index
from .retrieval.embedding import (
    EmbeddingService,
    MockEmbeddingService,
    SentenceTransformerEmbedding,
)
from .retrieval.engine import RetrievalEngine, RetrievalResult
from .session_buffer.buffer import SessionBuffer
from .session_buffer.models import BufferEntry
from .session_buffer.wal import WALManager
from .snapshots.manager import SnapshotManager
from .types import BufferEntryType, EdgeType, NodeStatus, NodeType, Role

__version__ = "0.2.0"

__all__ = [
    "AWHMConfig",
    "AWHMSession",
    "BM25Index",
    "BufferEntryType",
    "DeletionResult",
    "EdgeType",
    "EmbeddingService",
    "MemoryGraph",
    "MockEmbeddingService",
    "NodeStatus",
    "NodeType",
    "RetrievalEngine",
    "RetrievalResult",
    "Role",
    "SentenceTransformerEmbedding",
    "SessionBuffer",
    "SnapshotManager",
    "Stage1Pipeline",
    "StrengthScorer",
    "WALManager",
    "__version__",
]

logging.getLogger("awhm").addHandler(logging.NullHandler())

# Roles whose messages feed the real-time session buffer.
_BUFFERED_ROLES = frozenset({Role.USER.value, Role.TOOL_RESULT.value})


class AWHMSession:
    """Top-level facade: logging, buffering, retrieval, consolidation, deletion.

    Prefer :meth:`start_session` over the constructor; it loads the graph,
    wires the embedding service and recovers the write-ahead log. Sessions
    are context managers, so ``with AWHMSession.start_session() as s:`` ends
    the session (final flush, graph save) on exit.
    """

    def __init__(
        self,
        config: AWHMConfig,
        session_id: str,
        graph: MemoryGraph,
        embedding_service: EmbeddingService,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.graph = graph
        self.embedding = embedding_service

        self.logger = RawLogger(config, session_id)
        self.buffer = SessionBuffer()
        self.wal = WALManager(config, self.buffer, session_id)
        self.log_reader = RawLogReader(config)
        self.retrieval = self._build_retrieval_engine()
        self.snapshots = SnapshotManager(config)
        self._closed = False

    def _build_retrieval_engine(self) -> RetrievalEngine:
        return RetrievalEngine(
            self.config, self.graph, self.buffer, self.embedding, self.log_reader,
        )

    # ── Lifecycle ──────────────────────────────────────────────

    @classmethod
    def start_session(
        cls,
        config: AWHMConfig | None = None,
        session_id: str | None = None,
        use_mock_embeddings: bool = False,
    ) -> AWHMSession:
        """Create and start a session.

        Args:
            config: Configuration; defaults are used when ``None``.
            session_id: Custom session id; a UUID is generated when ``None``.
            use_mock_embeddings: Deterministic hash-based embeddings, for
                tests and for running without sentence-transformers.
        """
        config = config or AWHMConfig()
        config.ensure_dirs()
        session_id = session_id or str(uuid.uuid4())

        graph = load_graph(config)
        embedding: EmbeddingService
        if use_mock_embeddings:
            embedding = MockEmbeddingService(config.embed_dim)
        else:
            embedding = SentenceTransformerEmbedding(config.embed_model)

        session = cls(config, session_id, graph, embedding)
        session.wal.recover()  # crash recovery
        session.wal.start()
        return session

    def end_session(self) -> None:
        """End the session: stop the WAL timer, flush, and save the graph."""
        if self._closed:
            return
        self._closed = True
        self.wal.stop()
        self.wal.clear_wal()
        save_graph(self.graph, self.config)

    def __enter__(self) -> AWHMSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.end_session()

    # ── Logging and retrieval ──────────────────────────────────

    def log_message(
        self,
        role: Role | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to the raw log and run buffer pattern matching."""
        entry = self.logger.log(role, content, metadata)
        role_val = role.value if isinstance(role, Role) else role
        if role_val in _BUFFERED_ROLES:
            self.buffer.process_message(content, entry.timestamp, self.logger.msg_index - 1)

    def query(
        self,
        query_text: str,
        k: int | None = None,
        include_history: bool | None = None,
        with_trace: bool | None = None,
        as_of: str | datetime | None = None,
        semantic: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve the memories most relevant to ``query_text``.

        ``as_of`` answers as of a past moment (validity windows instead of
        current status); ``semantic=False`` skips embeddings entirely.
        """
        return self.retrieval.query(
            query_text,
            k=k,
            include_history=include_history,
            with_trace=with_trace,
            as_of=as_of,
            semantic=semantic,
        )

    # ── Consolidation ──────────────────────────────────────────

    def consolidate(self) -> dict[str, int]:
        """Run Stage 1 consolidation on every session with unprocessed messages."""
        return Stage1Pipeline(self.config, self.graph, self.embedding).consolidate_all_pending()

    def consolidate_current(self) -> int:
        """Consolidate this session's unprocessed messages."""
        return Stage1Pipeline(self.config, self.graph, self.embedding).consolidate_session(
            self.session_id,
        )

    # ── Deletion and snapshots ─────────────────────────────────

    def delete_node(self, node_id: str) -> DeletionResult:
        """Hard-delete a node and everything derived from it (privacy)."""
        return hard_delete(
            node_id, self.graph, self.config, self.buffer, current_session_id=self.session_id,
        )

    def create_snapshot(self) -> str:
        """Snapshot the graph and buffer. Returns the snapshot path."""
        return str(self.snapshots.create(self.graph, self.buffer.to_dicts()))

    def restore_snapshot(self, path: str) -> None:
        """Replace the in-memory graph and buffer with a snapshot's contents."""
        self.graph, wal_state = self.snapshots.restore(path)
        self.buffer.clear()
        for d in wal_state:
            self.buffer.add_entry(BufferEntry.from_dict(d))
        self.retrieval = self._build_retrieval_engine()
        save_graph(self.graph, self.config)
        self.wal.flush(force=True)

    # ── Introspection ──────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Summary counters for the memory system."""
        return {
            "session_id": self.session_id,
            "node_count": self.graph.node_count(),
            "edge_count": self.graph.edge_count(),
            "buffer_entries": len(self.buffer),
            "total_sessions": self.log_reader.session_count(),
            "log_messages": self.logger.msg_index,
        }
