"""AWHM Lite — External memory system for LLMs.

Top-level public API via AWHMSession facade.
"""

from __future__ import annotations

import uuid
from typing import Any

from .config import AWHMConfig
from .consolidation.pipeline import Stage1Pipeline
from .deletion.hard_delete import hard_delete, DeletionResult
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
from .session_buffer.wal import WALManager
from .snapshots.manager import SnapshotManager
from .types import BufferEntryType, EdgeType, NodeStatus, NodeType, Role


class AWHMSession:
    """Top-level facade for AWHM Lite.

    Usage:
        session = AWHMSession.start_session(config)
        session.log_message(Role.USER, "Hello")
        results = session.query("some query")
        session.end_session()
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

    def _build_retrieval_engine(self) -> RetrievalEngine:
        return RetrievalEngine(
            self.config, self.graph, self.buffer, self.embedding, self.log_reader,
        )

    @classmethod
    def start_session(
        cls,
        config: AWHMConfig | None = None,
        session_id: str | None = None,
        use_mock_embeddings: bool = False,
    ) -> AWHMSession:
        """Create and start a new session.

        Args:
            config: Configuration. Uses defaults if None.
            session_id: Custom session ID. Generated if None.
            use_mock_embeddings: Use mock embeddings for testing.
        """
        if config is None:
            config = AWHMConfig()
        config.ensure_dirs()

        if session_id is None:
            session_id = str(uuid.uuid4())

        # Load existing graph
        graph = load_graph(config)

        # Initialize embedding service
        if use_mock_embeddings:
            embedding: EmbeddingService = MockEmbeddingService(config.embed_dim)
        else:
            embedding = SentenceTransformerEmbedding(config.embed_model)

        session = cls(config, session_id, graph, embedding)

        # Recover WAL if exists (crash recovery)
        session.wal.recover()

        # Start WAL flushing
        session.wal.start()

        return session

    def log_message(
        self,
        role: Role | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a message and run buffer pattern matching."""
        entry = self.logger.log(role, content, metadata)

        # Run pattern matching on user messages
        role_val = role.value if isinstance(role, Role) else role
        if role_val in (Role.USER.value, Role.TOOL_RESULT.value):
            self.buffer.process_message(
                content, entry.timestamp, self.logger.msg_index - 1,
            )

    def query(
        self,
        query_text: str,
        k: int | None = None,
        include_history: bool | None = None,
        with_trace: bool | None = None,
    ) -> list[RetrievalResult]:
        """Query the memory system."""
        return self.retrieval.query(
            query_text,
            k=k,
            include_history=include_history,
            with_trace=with_trace,
        )

    def consolidate(self) -> dict[str, int]:
        """Run Stage 1 consolidation on all pending sessions."""
        pipeline = Stage1Pipeline(self.config, self.graph, self.embedding)
        return pipeline.consolidate_all_pending()

    def consolidate_current(self) -> int:
        """Consolidate the current session."""
        pipeline = Stage1Pipeline(self.config, self.graph, self.embedding)
        return pipeline.consolidate_session(self.session_id)

    def delete_node(self, node_id: str) -> DeletionResult:
        """Hard-delete a node for privacy compliance."""
        return hard_delete(
            node_id,
            self.graph,
            self.config,
            self.buffer,
            current_session_id=self.session_id,
        )

    def create_snapshot(self) -> str:
        """Create a snapshot. Returns the snapshot path."""
        path = self.snapshots.create(self.graph, self.buffer.to_dicts())
        return str(path)

    def restore_snapshot(self, path: str) -> None:
        """Restore from a snapshot."""
        self.graph, wal_state = self.snapshots.restore(path)
        self.buffer.clear()
        from .session_buffer.models import BufferEntry
        for d in wal_state:
            self.buffer.add_entry(BufferEntry.from_dict(d))
        # Rebind retrieval to restored in-memory graph state.
        self.retrieval = self._build_retrieval_engine()
        save_graph(self.graph, self.config)
        self.wal.flush()

    def status(self) -> dict[str, Any]:
        """Return status info about the memory system."""
        return {
            "session_id": self.session_id,
            "node_count": self.graph.node_count(),
            "edge_count": self.graph.edge_count(),
            "buffer_entries": len(self.buffer.entries),
            "total_sessions": self.log_reader.session_count(),
            "log_messages": self.logger.msg_index,
        }

    def end_session(self) -> None:
        """End the session: stop WAL, flush final state."""
        self.wal.stop()
        self.wal.clear_wal()
        save_graph(self.graph, self.config)
