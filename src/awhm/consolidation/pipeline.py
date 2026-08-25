"""Stage1Pipeline: orchestrates full consolidation (zero LLM calls)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import AWHMConfig
from ..graph.memory_graph import MemoryGraph
from ..graph.models import MemoryEdge, MemoryNode, StrengthScore
from ..graph.serialization import save_graph
from ..graph.strength import StrengthScorer
from ..raw_log.reader import RawLogReader
from ..retrieval.embedding import EmbeddingService
from ..types import EdgeType, NodeStatus, NodeType
from .canonical import canonical_key_for_content, normalize_text
from .deduplication import find_duplicates
from .entity_linking import find_links
from .extraction import extract_from_log_entries
from .ner import NERExtractor
from .temporal import TemporalParser


class Stage1Pipeline:
    """Symbolic consolidation pipeline (spec section 5).

    Pipeline:
    1. NER — extract entities from raw logs
    2. Temporal parsing — resolve dates/times
    3. Rule-based extraction — corrections, preferences, facts, outcomes
    4. Entity linking — match to existing graph nodes
    5. Deduplication — detect near-duplicates
    6. Commit — add new nodes and edges
    """

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
        self.ner = NERExtractor()
        self.temporal = TemporalParser()
        self.scorer = StrengthScorer(config)

    def consolidate_session(
        self,
        session_id: str,
        _state: dict[str, dict[str, Any]] | None = None,
    ) -> int:
        """Run full consolidation for a single session. Returns count of new nodes."""
        entries = self.log_reader.read_session(session_id)
        if not entries:
            return 0

        state = _state if _state is not None else self._load_state()
        last_processed = self._last_processed_index(state, session_id)
        if last_processed > len(entries) - 1:
            # Log may have been truncated (e.g., privacy deletions). Clamp progress.
            self._set_last_processed_index(state, session_id, len(entries) - 1)
            return 0
        if last_processed == len(entries) - 1:
            return 0

        start_idx = max(last_processed + 1, 0)
        new_entries = entries[start_idx:]
        messages = [e.content for e in new_entries]
        now = datetime.now(timezone.utc)
        new_node_count = 0

        # 1. NER
        try:
            entities = self.ner.extract_from_messages(
                messages, message_offset=start_idx,
            )
        except RuntimeError:
            entities = []

        # 2. Temporal parsing
        try:
            dates = self.temporal.extract_from_messages(
                messages, message_offset=start_idx,
            )
        except Exception:
            dates = []

        # 3. Rule-based extraction
        extractions = extract_from_log_entries(
            new_entries, message_offset=start_idx,
        )

        # Collect all new content candidates
        new_contents: list[str] = []
        content_metadata: list[dict[str, Any]] = []

        # From NER entities -> semantic nodes
        for ent in entities:
            content = f"{ent.label}: {ent.text}"
            new_contents.append(content)
            refs = []
            if ent.message_index is not None:
                refs.append({
                    "session_id": session_id,
                    "message_index": ent.message_index,
                })
            content_metadata.append({
                "type": NodeType.SEMANTIC.value,
                "source": "ner",
                "entity_text": ent.text,
                "entity_label": ent.label,
                "confidence": 0.75,
                "refs": refs,
            })

        # From extractions -> appropriate node types
        for ext in extractions:
            new_contents.append(ext.content)
            if ext.type in ("correction", "fact"):
                ntype = NodeType.SEMANTIC.value
            elif ext.type == "preference":
                ntype = NodeType.PROCEDURAL.value
            else:
                ntype = NodeType.EPISODIC.value
            content_metadata.append({
                "type": ntype,
                "source": "extraction",
                "extraction_type": ext.type,
                "confidence": self._confidence_for_extraction_type(ext.type),
                "refs": [{
                    "session_id": session_id,
                    "message_index": ext.message_index,
                }],
            })

        # From temporal -> episodic nodes
        for d in dates:
            content = f"Date reference: {d.original} -> {d.iso}"
            new_contents.append(content)
            refs = []
            if d.message_index is not None:
                refs.append({
                    "session_id": session_id,
                    "message_index": d.message_index,
                })
            content_metadata.append({
                "type": NodeType.EPISODIC.value,
                "source": "temporal",
                "confidence": 0.65,
                "refs": refs,
            })

        if not new_contents:
            self._set_last_processed_index(state, session_id, len(entries) - 1)
            return 0

        refs_by_content: dict[str, list[dict[str, Any]]] = {}
        for i, content in enumerate(new_contents):
            refs = content_metadata[i].get("refs", [])
            if not refs:
                continue
            refs_by_content.setdefault(content, []).extend(refs)

        # Embed all new content
        new_embeddings = self.embedding.encode(new_contents)

        # 4. Entity linking
        entity_texts = [ent.text for ent in entities]
        entity_labels = [ent.label for ent in entities]
        links = find_links(
            entity_texts, self.graph, self.embedding,
            threshold=self.config.entity_link_threshold,
            entity_labels=entity_labels,
        )
        linked_texts = {lc.entity_text.lower() for lc in links}

        # 5. Deduplication
        duplicates = find_duplicates(
            new_contents, new_embeddings, self.graph,
            threshold=self.config.dedup_threshold,
        )
        dup_contents = {dc.new_content for dc in duplicates}

        # For duplicates, update existing node's session list and access count
        for dc in duplicates:
            existing = self.graph.get_node(dc.existing_node_id)
            if existing and session_id not in existing.source_sessions:
                existing.source_sessions.append(session_id)
                existing.access_count += 1
            if existing:
                refs = refs_by_content.get(dc.new_content, [])
                self._merge_source_refs(existing, refs)

        # 6. Commit — add non-duplicate, non-linked-entity nodes
        prev_node_id: str | None = None
        created_nodes_by_entity: dict[str, list[str]] = {}
        for i, content in enumerate(new_contents):
            if content in dup_contents:
                continue

            # Skip NER entities already linked to existing nodes
            meta = content_metadata[i]
            if meta["source"] == "ner":
                ent_text = content.split(": ", 1)[-1] if ": " in content else content
                if ent_text.lower() in linked_texts:
                    continue

            canonical_key = canonical_key_for_content(content, meta["type"])
            superseded_ids = self._supersede_active_nodes(
                canonical_key,
                content,
                now_iso=now.isoformat(),
            )
            node = MemoryNode(
                id=str(uuid.uuid4()),
                type=meta["type"],
                content=content,
                embedding=new_embeddings[i].tolist(),
                embed_model=self.config.embed_model,
                strength=StrengthScore(recency=1.0, frequency=1, composite=1.0),
                created_at=now.isoformat(),
                last_accessed=now.isoformat(),
                source_sessions=[session_id],
                source_refs=meta.get("refs", []),
                access_count=1,
                canonical_key=canonical_key,
                status=NodeStatus.ACTIVE.value,
                supersedes=superseded_ids,
                valid_from=now.isoformat(),
                valid_to=None,
                confidence=float(meta.get("confidence", 0.6)),
                entity_type=meta.get("entity_label"),
            )
            self.graph.add_node(node)
            new_node_count += 1

            if meta["source"] == "ner":
                entity_text = str(meta.get("entity_text", "")).strip().lower()
                if entity_text:
                    created_nodes_by_entity.setdefault(entity_text, []).append(node.id)

            # Temporal edges between sequential episodic nodes
            if prev_node_id and meta["type"] == NodeType.EPISODIC.value:
                edge = MemoryEdge(
                    source=prev_node_id,
                    target=node.id,
                    type=EdgeType.TEMPORAL.value,
                    weight=1.0,
                    created_at=now.isoformat(),
                )
                self.graph.add_edge(edge)

            if meta["type"] == NodeType.EPISODIC.value:
                prev_node_id = node.id

        # Add association edges from linked entities
        for lc in links:
            source_ids = created_nodes_by_entity.get(lc.entity_text.lower(), [])
            for source_id in source_ids:
                if source_id == lc.matched_node_id:
                    continue
                edge = MemoryEdge(
                    source=source_id,
                    target=lc.matched_node_id,
                    type=EdgeType.ASSOCIATION.value,
                    weight=lc.embedding_sim,
                    created_at=now.isoformat(),
                )
                self.graph.add_edge(edge)

        # Update strength scores
        self.scorer.update_all(self.graph, now)

        # Save graph
        save_graph(self.graph, self.config)

        # Mark this session progress as fully processed at current log tail.
        self._set_last_processed_index(state, session_id, len(entries) - 1)

        return new_node_count

    def consolidate_all_pending(self) -> dict[str, int]:
        """Consolidate all sessions with unprocessed log entries."""
        state = self._load_state()
        all_sessions = self.log_reader.list_sessions()

        results: dict[str, int] = {}
        for sid in all_sessions:
            entries = self.log_reader.read_session(sid)
            if not entries:
                continue
            last_processed = self._last_processed_index(state, sid)
            if last_processed > len(entries) - 1:
                self._set_last_processed_index(state, sid, len(entries) - 1)
                continue
            if last_processed == len(entries) - 1:
                continue
            results[sid] = self.consolidate_session(sid, _state=state)
        return results

    def _load_state(self) -> dict[str, dict[str, Any]]:
        """Load consolidation progress state with legacy migration support."""
        path = self.config.consolidated_sessions_path
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sessions_raw = data.get("sessions", {})

        # Legacy schema: {"sessions": ["sid1", "sid2", ...]}
        if isinstance(sessions_raw, list):
            now_iso = datetime.now(timezone.utc).isoformat()
            migrated: dict[str, dict[str, Any]] = {}
            for sid in sessions_raw:
                entry_count = len(self.log_reader.read_session(sid))
                migrated[sid] = {
                    "last_msg_index": max(entry_count - 1, -1),
                    "updated_at": now_iso,
                }
            self._save_state(migrated)
            return migrated

        sessions: dict[str, dict[str, Any]] = {}
        changed = False
        if isinstance(sessions_raw, dict):
            for sid, value in sessions_raw.items():
                if isinstance(value, dict):
                    last = int(value.get("last_msg_index", -1))
                    updated = value.get("updated_at")
                    if not isinstance(updated, str):
                        updated = datetime.now(timezone.utc).isoformat()
                        changed = True
                    sessions[sid] = {
                        "last_msg_index": last,
                        "updated_at": updated,
                    }
                else:
                    # Legacy-adjacent schema: {"sessions": {"sid": 12}}
                    sessions[sid] = {
                        "last_msg_index": int(value),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    changed = True

        if changed:
            self._save_state(sessions)
        return sessions

    def _save_state(self, sessions: dict[str, dict[str, Any]]) -> None:
        self.config.ensure_dirs()
        path = self.config.consolidated_sessions_path
        data = {
            "version": 2,
            "sessions": sessions,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def _last_processed_index(
        self,
        state: dict[str, dict[str, Any]],
        session_id: str,
    ) -> int:
        rec = state.get(session_id, {})
        return int(rec.get("last_msg_index", -1))

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
    def _confidence_for_extraction_type(extraction_type: str) -> float:
        if extraction_type == "correction":
            return 0.9
        if extraction_type == "fact":
            return 0.8
        if extraction_type == "preference":
            return 0.75
        return 0.65

    def _supersede_active_nodes(
        self,
        canonical_key: str | None,
        new_content: str,
        now_iso: str,
    ) -> list[str]:
        """Mark older active nodes as superseded when a canonical key collides."""
        if not canonical_key:
            return []
        new_norm = normalize_text(new_content)
        superseded: list[str] = []
        for node in self.graph.nodes.values():
            if node.status != NodeStatus.ACTIVE.value:
                continue
            if node.canonical_key != canonical_key:
                continue
            if normalize_text(node.content) == new_norm:
                continue
            node.status = NodeStatus.SUPERSEDED.value
            node.valid_to = now_iso
            superseded.append(node.id)
        return superseded

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
            if key in seen:
                continue
            seen.add(key)
            node.source_refs.append(ref)
