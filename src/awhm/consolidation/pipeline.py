"""Stage1Pipeline: symbolic consolidation from raw logs into the memory graph.

Zero LLM calls. For each session with unprocessed messages:

1. NER (spaCy) and temporal parsing (dateparser) over the new messages
2. Rule-based extraction of corrections, preferences, facts and outcomes
3. Entity linking against existing nodes
4. Near-duplicate detection against existing nodes
5. Commit: assign canonical keys, supersede contradicted memories, add
   nodes and edges, refresh strength scores, persist
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import AWHMConfig
from ..graph.memory_graph import MemoryGraph
from ..graph.models import MemoryEdge, MemoryNode, StrengthScore
from ..graph.serialization import save_graph
from ..graph.strength import StrengthScorer
from ..raw_log.models import LogEntry
from ..raw_log.reader import RawLogReader
from ..retrieval.embedding import EmbeddingService
from ..types import EdgeType, NodeStatus, NodeType
from .canonical import canonical_key_for_content, correction_supersedes, is_correction
from .deduplication import find_duplicates, first_occurrences, statement_key
from .entity_linking import find_links
from .extraction import extract_from_log_entries
from .ner import NERExtractor
from .temporal import TemporalParser

logger = logging.getLogger("awhm.consolidation")

CONFIDENCE_BY_EXTRACTION = {
    "correction": 0.9,
    "fact": 0.8,
    "preference": 0.75,
    "outcome": 0.65,
}
CONFIDENCE_NER = 0.75
CONFIDENCE_TEMPORAL = 0.65

NODE_TYPE_BY_EXTRACTION = {
    "correction": NodeType.SEMANTIC.value,
    "fact": NodeType.SEMANTIC.value,
    "preference": NodeType.PROCEDURAL.value,
    "outcome": NodeType.EPISODIC.value,
}


@dataclass
class Candidate:
    """A piece of content proposed for the graph, with its provenance."""

    content: str
    node_type: str
    source: str  # "ner" | "extraction" | "temporal"
    confidence: float
    message_index: int | None
    session_id: str
    entity_text: str | None = None
    entity_label: str | None = None
    extraction_type: str | None = None
    is_correction: bool = False
    refs: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.refs and self.message_index is not None:
            self.refs = [{"session_id": self.session_id, "message_index": self.message_index}]


class Stage1Pipeline:
    """Symbolic consolidation pipeline."""

    def __init__(
        self,
        config: AWHMConfig,
        graph: MemoryGraph,
        embedding_service: EmbeddingService,
    ) -> None:
        self.config = config
        self.graph = graph
        self.embedding = embedding_service
        self.log_reader = RawLogReader(config)
        self.ner = NERExtractor(labels=config.ner_labels)
        self.temporal = TemporalParser()
        self.scorer = StrengthScorer(config)
        self._ner_unavailable_logged = False

    # ── Public API ─────────────────────────────────────────────

    def consolidate_session(
        self,
        session_id: str,
        _state: dict[str, dict[str, Any]] | None = None,
    ) -> int:
        """Consolidate unprocessed messages of one session. Returns new node count."""
        entries = self.log_reader.read_session(session_id)
        if not entries:
            return 0

        state = _state if _state is not None else self._load_state()
        last_processed = self._last_processed_index(state, session_id)
        tail = len(entries) - 1
        if last_processed >= tail:
            if last_processed > tail:
                # The log shrank (privacy deletion); clamp progress to the new tail.
                self._set_last_processed_index(state, session_id, tail)
            return 0

        start_idx = last_processed + 1
        new_entries = entries[start_idx:]
        now = datetime.now(timezone.utc)

        candidates = self._gather_candidates(session_id, new_entries, start_idx)
        if not candidates:
            self._set_last_processed_index(state, session_id, tail)
            return 0

        new_count = self._commit(session_id, candidates, now)

        self.scorer.update_all(self.graph, now)
        save_graph(self.graph, self.config)
        self._set_last_processed_index(state, session_id, tail)
        logger.info("consolidated session %s: %d new node(s)", session_id, new_count)
        return new_count

    def consolidate_all_pending(self) -> dict[str, int]:
        """Consolidate every session that has unprocessed log entries."""
        state = self._load_state()
        results: dict[str, int] = {}
        for sid in self.log_reader.list_sessions():
            entries = self.log_reader.read_session(sid)
            if not entries:
                continue
            tail = len(entries) - 1
            last_processed = self._last_processed_index(state, sid)
            if last_processed >= tail:
                if last_processed > tail:
                    self._set_last_processed_index(state, sid, tail)
                continue
            results[sid] = self.consolidate_session(sid, _state=state)
        return results

    # ── Candidate gathering ────────────────────────────────────

    def _gather_candidates(
        self,
        session_id: str,
        entries: list[LogEntry],
        start_idx: int,
    ) -> list[Candidate]:
        messages = [e.content for e in entries]
        candidates: list[Candidate] = []

        for ent in self._extract_entities(messages, start_idx):
            candidates.append(Candidate(
                content=f"{ent.label}: {ent.text}",
                node_type=NodeType.SEMANTIC.value,
                source="ner",
                confidence=CONFIDENCE_NER,
                message_index=ent.message_index,
                session_id=session_id,
                entity_text=ent.text,
                entity_label=ent.label,
            ))

        for ext in extract_from_log_entries(entries, message_offset=start_idx):
            candidates.append(Candidate(
                content=ext.content,
                node_type=NODE_TYPE_BY_EXTRACTION.get(ext.type, NodeType.EPISODIC.value),
                source="extraction",
                confidence=CONFIDENCE_BY_EXTRACTION.get(ext.type, 0.6),
                message_index=ext.message_index,
                session_id=session_id,
                extraction_type=ext.type,
                is_correction=ext.type == "correction" or is_correction(ext.content),
            ))

        for d in self._extract_dates(messages, start_idx):
            candidates.append(Candidate(
                content=f"Date reference: {d.original} -> {d.iso}",
                node_type=NodeType.EPISODIC.value,
                source="temporal",
                confidence=CONFIDENCE_TEMPORAL,
                message_index=d.message_index,
                session_id=session_id,
            ))

        keep = first_occurrences([c.content for c in candidates])
        return [candidates[i] for i in keep]

    def _extract_entities(self, messages: list[str], start_idx: int):
        try:
            return self.ner.extract_from_messages(messages, message_offset=start_idx)
        except RuntimeError as exc:
            if not self._ner_unavailable_logged:
                logger.warning("entity extraction disabled: %s", exc)
                self._ner_unavailable_logged = True
            return []

    def _extract_dates(self, messages: list[str], start_idx: int):
        try:
            return self.temporal.extract_from_messages(messages, message_offset=start_idx)
        except Exception as exc:  # dateparser can raise on exotic locale input
            logger.warning("temporal parsing failed: %s", exc)
            return []

    # ── Commit ─────────────────────────────────────────────────

    def _commit(self, session_id: str, candidates: list[Candidate], now: datetime) -> int:
        now_iso = now.isoformat()
        contents = [c.content for c in candidates]
        embeddings = self.embedding.encode(contents)

        # Entity linking: NER candidates that match an existing node are not
        # re-added; instead the new context gets an association edge to it.
        ner_candidates = [c for c in candidates if c.source == "ner"]
        links = find_links(
            [c.entity_text or "" for c in ner_candidates],
            self.graph,
            self.embedding,
            threshold=self.config.entity_link_threshold,
            entity_labels=[c.entity_label or "" for c in ner_candidates],
        )
        linked_entities = {lc.entity_text.lower() for lc in links}

        # Near-duplicates reinforce the existing node instead of creating a new one.
        duplicates = find_duplicates(
            contents, embeddings, self.graph, threshold=self.config.dedup_threshold,
        )
        duplicate_contents: set[str] = set()
        for dc in duplicates:
            duplicate_contents.add(dc.new_content)
            existing = self.graph.get_node(dc.existing_node_id)
            if existing is None:
                continue
            if session_id not in existing.source_sessions:
                existing.source_sessions.append(session_id)
                existing.access_count += 1
            refs = [c.refs for c in candidates if c.content == dc.new_content]
            self._merge_source_refs(existing, [r for rs in refs for r in rs])

        new_count = 0
        prev_episodic_id: str | None = None
        created_by_entity: dict[str, list[str]] = {}

        for i, cand in enumerate(candidates):
            if cand.content in duplicate_contents:
                continue
            if cand.source == "ner" and (cand.entity_text or "").lower() in linked_entities:
                continue

            key = canonical_key_for_content(cand.content, cand.node_type)
            superseded = self._supersede(cand, key, now_iso)

            node = MemoryNode(
                id=str(uuid.uuid4()),
                type=cand.node_type,
                content=cand.content,
                embedding=embeddings[i].tolist(),
                embed_model=self.config.embed_model,
                strength=StrengthScore(recency=1.0, frequency=1, composite=1.0),
                created_at=now_iso,
                last_accessed=now_iso,
                source_sessions=[session_id],
                source_refs=list(cand.refs),
                access_count=1,
                canonical_key=key,
                status=NodeStatus.ACTIVE.value,
                supersedes=superseded,
                valid_from=now_iso,
                valid_to=None,
                confidence=cand.confidence,
                entity_type=cand.entity_label,
            )
            self.graph.add_node(node)
            new_count += 1

            if cand.source == "ner" and cand.entity_text:
                created_by_entity.setdefault(cand.entity_text.lower(), []).append(node.id)

            if cand.node_type == NodeType.EPISODIC.value:
                if prev_episodic_id is not None:
                    self.graph.add_edge(MemoryEdge(
                        source=prev_episodic_id,
                        target=node.id,
                        type=EdgeType.TEMPORAL.value,
                        weight=1.0,
                        created_at=now_iso,
                    ))
                prev_episodic_id = node.id

        for lc in links:
            for source_id in created_by_entity.get(lc.entity_text.lower(), []):
                if source_id != lc.matched_node_id:
                    self.graph.add_edge(MemoryEdge(
                        source=source_id,
                        target=lc.matched_node_id,
                        type=EdgeType.ASSOCIATION.value,
                        weight=lc.embedding_sim,
                        created_at=now_iso,
                    ))

        return new_count

    def _supersede(self, cand: Candidate, key: str | None, now_iso: str) -> list[str]:
        """Mark active nodes contradicted by ``cand`` as superseded. Returns their ids."""
        if key is None:
            return []
        new_statement = statement_key(cand.content)
        superseded: list[str] = []
        for node in self.graph.active_nodes():
            if node.canonical_key is None or statement_key(node.content) == new_statement:
                continue  # a restatement is not a contradiction
            distance = self._message_distance(cand, node)
            if correction_supersedes(
                key,
                cand.is_correction,
                node.canonical_key,
                distance,
                self.config.correction_window_messages,
            ):
                node.status = NodeStatus.SUPERSEDED.value
                node.valid_to = now_iso
                superseded.append(node.id)
        return superseded

    @staticmethod
    def _message_distance(cand: Candidate, node: MemoryNode) -> int | None:
        """Messages between ``node``'s latest source in this session and ``cand``."""
        if cand.message_index is None:
            return None
        same_session = [
            int(ref["message_index"])
            for ref in node.source_refs
            if isinstance(ref, dict)
            and ref.get("session_id") == cand.session_id
            and ref.get("message_index") is not None
        ]
        if not same_session:
            return None
        return cand.message_index - max(same_session)

    # ── Progress state ─────────────────────────────────────────

    def _load_state(self) -> dict[str, dict[str, Any]]:
        """Load consolidation progress, migrating older schemas in place."""
        path = self.config.consolidated_sessions_path
        if not path.exists():
            return {}

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sessions_raw = data.get("sessions", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        # v1 schema: {"sessions": ["sid1", "sid2"]} meant "fully processed".
        if isinstance(sessions_raw, list):
            migrated = {
                sid: {
                    "last_msg_index": len(self.log_reader.read_session(sid)) - 1,
                    "updated_at": now_iso,
                }
                for sid in sessions_raw
            }
            self._save_state(migrated)
            return migrated

        sessions: dict[str, dict[str, Any]] = {}
        changed = False
        if isinstance(sessions_raw, dict):
            for sid, value in sessions_raw.items():
                if isinstance(value, dict):
                    updated = value.get("updated_at")
                    if not isinstance(updated, str):
                        updated, changed = now_iso, True
                    sessions[sid] = {
                        "last_msg_index": int(value.get("last_msg_index", -1)),
                        "updated_at": updated,
                    }
                else:  # {"sid": 12}
                    sessions[sid] = {"last_msg_index": int(value), "updated_at": now_iso}
                    changed = True
        if changed:
            self._save_state(sessions)
        return sessions

    def _save_state(self, sessions: dict[str, dict[str, Any]]) -> None:
        self.config.ensure_dirs()
        path = self.config.consolidated_sessions_path
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "sessions": sessions}, f, indent=2)
        tmp.replace(path)

    @staticmethod
    def _last_processed_index(state: dict[str, dict[str, Any]], session_id: str) -> int:
        return int(state.get(session_id, {}).get("last_msg_index", -1))

    def _set_last_processed_index(
        self,
        state: dict[str, dict[str, Any]],
        session_id: str,
        last_msg_index: int,
    ) -> None:
        state[session_id] = {
            "last_msg_index": last_msg_index,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state(state)

    @staticmethod
    def _merge_source_refs(node: MemoryNode, new_refs: list[dict[str, Any]]) -> None:
        seen = {
            (ref.get("session_id"), ref.get("message_index"))
            for ref in node.source_refs
            if isinstance(ref, dict)
        }
        for ref in new_refs:
            if not isinstance(ref, dict):
                continue
            key = (ref.get("session_id"), ref.get("message_index"))
            if key not in seen:
                seen.add(key)
                node.source_refs.append(ref)
