"""Configuration dataclass with all tunable parameters and path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

DEFAULT_DATA_DIR = "~/.awhm"

# spaCy entity labels worth remembering. Numeric and time-like labels
# (CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY, TIME, DATE) produce noise
# nodes; DATE is handled separately by the temporal parser.
DEFAULT_NER_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LANGUAGE", "NORP", "FAC", "LAW",
})


@dataclass
class AWHMConfig:
    """All tunable parameters plus path helpers.

    ``data_dir`` accepts ``~`` and environment variables; it is expanded on
    construction so every consumer sees an absolute path.
    """

    data_dir: str = DEFAULT_DATA_DIR

    # Strength scoring: recency decay s_rec = (1 + beta * hours) ** (-alpha)
    alpha: float = 0.3
    beta: float = 0.1

    # Strength scoring: composite weights S = w_rec * s_rec + w_freq * s_freq
    w_rec: float = 0.4
    w_freq: float = 0.6

    # Legacy two-signal ranking (kept for backward compatibility)
    w_sim: float = 0.7
    w_str: float = 0.3
    eta: float = 0.5

    # Retrieval feature blend
    retrieval_profile: str = "balanced"
    w_semantic: float = 0.55
    w_lexical: float = 0.20
    w_strength: float = 0.15
    w_confidence: float = 0.10
    w_association: float = 0.10       # Weight of graph-neighbour evidence
    contradiction_penalty: float = 0.35
    include_history_by_default: bool = False
    trace_retrieval: bool = False

    # Retrieval
    k: int = 10
    bm25_anchor_ratio: float = 0.5      # Lexical anchor: score >= ratio * best BM25 score
    embed_threshold: float = 0.3        # Minimum cosine similarity for a semantic anchor
    raw_log_score_scale: float = 0.5    # Cold-start hits are scaled into [0, scale]
    neighbor_expansion: bool = True     # Pull in one-hop graph neighbours of anchors
    neighbor_decay: float = 0.6         # Edge weight multiplier for expanded neighbours

    # Consolidation
    entity_link_threshold: float = 0.85     # Cosine threshold for entity linking
    dedup_threshold: float = 0.92           # Cosine threshold for near-duplicates
    correction_window_messages: int = 3   # A correction supersedes a same-family statement this close
    ner_labels: frozenset[str] = DEFAULT_NER_LABELS

    # Session buffer
    buffer_flush_interval: float = 30.0  # WAL flush interval in seconds

    # Embedding model
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Cold-start: fall back to BM25 over raw logs while few sessions exist
    cold_start_session_count: int = 10

    # Stage 2: optional offline LLM refinement (needs an LLM client)
    stage2_enabled: bool = False
    stage2_model: str = "claude-opus-5"
    stage2_max_messages: int = 60       # transcript messages per LLM call
    stage2_min_confidence: float = 0.5  # drop proposals below this
    ann_index_type: str = "none"

    # Privacy/deletion behaviour
    delete_snapshots_on_hard_delete: bool = True

    # Storage backend for the graph: "json" (one file) or "sqlite" (incremental)
    storage_backend: str = "json"

    def __post_init__(self) -> None:
        self.data_dir = os.path.expanduser(os.path.expandvars(self.data_dir))
        self.ner_labels = frozenset(label.upper() for label in self.ner_labels)

    # ── Paths ──────────────────────────────────────────────────

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
    def sqlite_path(self) -> Path:
        return self.graph_dir / "memory_graph.sqlite"

    @property
    def snapshots_dir(self) -> Path:
        return self.base_path / "snapshots"

    @property
    def wal_dir(self) -> Path:
        return self.base_path / "wal"

    @property
    def wal_path(self) -> Path:
        return self.wal_dir / "session_buffer.wal"

    def log_path_for_session(self, session_id: str) -> Path:
        """Raw log file for a session; the id is percent-encoded so any string is safe."""
        return self.logs_dir / f"{quote(session_id, safe='')}.jsonl"

    @staticmethod
    def session_id_from_log_path(path: Path) -> str:
        return unquote(path.stem)

    def wal_path_for_session(self, session_id: str) -> Path:
        """Return session-scoped WAL path with a filesystem-safe filename."""
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
        for d in (self.logs_dir, self.graph_dir, self.snapshots_dir,
                  self.wal_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)
