"""Configuration dataclass with all spec parameters and path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AWHMConfig:
    """All 13 spec parameters + path helpers."""

    data_dir: str = field(default_factory=lambda: os.path.expanduser("~/.awhm"))

    # Strength scoring - recency decay
    alpha: float = 0.3          # Power-law exponent
    beta: float = 0.1           # Decay scaling constant

    # Strength scoring - composite weights
    w_rec: float = 0.4          # Recency weight
    w_freq: float = 0.6         # Frequency weight

    # Retrieval ranking
    w_sim: float = 0.7          # Similarity weight in R(v,q)
    w_str: float = 0.3          # Strength weight in R(v,q)
    eta: float = 0.5            # Similarity-gate exponent

    # Retrieval feature profile (new path; w_sim/w_str retained for compatibility)
    retrieval_profile: str = "balanced"
    w_semantic: float = 0.55
    w_lexical: float = 0.20
    w_strength: float = 0.15
    w_confidence: float = 0.10
    contradiction_penalty: float = 0.35
    include_history_by_default: bool = False
    trace_retrieval: bool = False

    # Retrieval
    k: int = 10                 # Top-k retrieval count
    bm25_threshold: float = 1.0         # Minimum BM25 score for anchor set
    embed_threshold: float = 0.3        # Minimum cosine sim for anchor set

    # Consolidation
    entity_link_threshold: float = 0.85  # Cosine threshold for entity linking
    dedup_threshold: float = 0.92        # Cosine threshold for deduplication

    # Session buffer
    buffer_flush_interval: float = 30.0  # WAL flush interval in seconds

    # Embedding model
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Cold-start
    cold_start_session_count: int = 10

    # Stage 2 (future-facing; optional)
    stage2_enabled: bool = False
    ann_index_type: str = "none"

    # Privacy/deletion behavior
    delete_snapshots_on_hard_delete: bool = True

    @property
    def base_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def logs_dir(self) -> Path:
        return self.base_path / "logs"

    @property
    def graph_dir(self) -> Path:
        return self.base_path / "graph"

    @property
    def graph_path(self) -> Path:
        return self.graph_dir / "memory_graph.json"

    @property
    def snapshots_dir(self) -> Path:
        return self.base_path / "snapshots"

    @property
    def wal_dir(self) -> Path:
        return self.base_path / "wal"

    @property
    def wal_path(self) -> Path:
        return self.wal_dir / "session_buffer.wal"

    def wal_path_for_session(self, session_id: str) -> Path:
        """Return session-scoped WAL path, with a filesystem-safe filename."""
        safe = "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_"
            for c in session_id
        )
        return self.wal_dir / f"{safe}.wal"

    @property
    def meta_dir(self) -> Path:
        return self.base_path / "meta"

    @property
    def consolidated_sessions_path(self) -> Path:
        return self.meta_dir / "consolidated_sessions.json"

    @property
    def deletion_tombstones_path(self) -> Path:
        return self.meta_dir / "deletion_tombstones.jsonl"

    @property
    def deletion_ledger_path(self) -> Path:
        return self.meta_dir / "deletion_ledger.jsonl"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [self.logs_dir, self.graph_dir, self.snapshots_dir,
                  self.wal_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)
