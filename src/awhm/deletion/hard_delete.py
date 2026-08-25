"""Hard deletion: cascade a node's removal across graph, logs, buffer and snapshots."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import AWHMConfig
from ..graph.memory_graph import MemoryGraph
from ..graph.serialization import save_graph
from ..raw_log.reader import RawLogReader
from ..session_buffer.buffer import SessionBuffer


@dataclass
class DeletionResult:
    node_deleted: bool
    edges_removed: int
    log_entries_removed: int
    buffer_entries_removed: int
    affected_sessions: list[str]
    match_strategy: str
    exact_matches_removed: int
    snapshots_touched: int
    tombstone_id: str | None
    ledger_record_id: str | None


def hard_delete(
    node_id: str,
    graph: MemoryGraph,
    config: AWHMConfig,
    buffer: SessionBuffer | None = None,
    current_session_id: str | None = None,
) -> DeletionResult:
    """Hard-delete a node and everything derived from or feeding it.

    1. Remove the node and its incident edges from the graph
    2. Remove the raw log messages it was extracted from (by source refs when
       available, else by exact content match)
    3. Remove matching session-buffer entries
    4. Scrub the node from existing snapshots
    5. Record a tombstone and a ledger entry for auditability
    """
    result = DeletionResult(
        node_deleted=False,
        edges_removed=0,
        log_entries_removed=0,
        buffer_entries_removed=0,
        affected_sessions=[],
        match_strategy="none",
        exact_matches_removed=0,
        snapshots_touched=0,
        tombstone_id=None,
        ledger_record_id=None,
    )

    node = graph.get_node(node_id)
    if node is None:
        return result

    content = node.content
    source_sessions = list(node.source_sessions)
    source_refs = _normalize_source_refs(node.source_refs)

    result.edges_removed = len(graph.get_edges_for_node(node_id))
    result.node_deleted = graph.remove_node(node_id) is not None

    # Raw logs
    reader = RawLogReader(config)
    sessions_affected: set[str] = set()
    refs_by_session: dict[str, set[int]] = {}
    for ref in source_refs:
        refs_by_session.setdefault(str(ref["session_id"]), set()).add(int(ref["message_index"]))

    if refs_by_session:
        result.match_strategy = "source_refs"
        for session_id, indices in refs_by_session.items():
            count = reader.hard_delete_entries_by_indices(session_id, indices)
            if count:
                sessions_affected.add(session_id)
            result.log_entries_removed += count
    else:
        result.match_strategy = "exact_content"
        for session_id in source_sessions:
            count = reader.hard_delete_entries_exact(session_id, content)
            if count:
                sessions_affected.add(session_id)
            result.log_entries_removed += count
        result.exact_matches_removed = result.log_entries_removed

    # Session buffer
    if buffer is not None:
        source_msg_ids = {
            int(ref["message_index"])
            for ref in source_refs
            if current_session_id and ref["session_id"] == current_session_id
        }
        if source_msg_ids:
            result.buffer_entries_removed = buffer.remove_where(
                lambda e: e.source_msg in source_msg_ids
            )
        else:
            target = _normalize_text(content)
            result.buffer_entries_removed = buffer.remove_where(
                lambda e: _normalize_text(e.content) == target
            )

    result.affected_sessions = sorted(sessions_affected)

    if config.delete_snapshots_on_hard_delete:
        result.snapshots_touched = _scrub_snapshots(config, node_id, content)

    stamp = datetime.now(timezone.utc).isoformat()
    result.tombstone_id = str(uuid.uuid4())
    result.ledger_record_id = str(uuid.uuid4())
    _append_jsonl(config.deletion_tombstones_path, {
        "tombstone_id": result.tombstone_id,
        "timestamp": stamp,
        "node_id": node_id,
        "content_hash_hint": _normalize_text(content)[:64],
        "affected_sessions": result.affected_sessions,
        "match_strategy": result.match_strategy,
    })
    _append_jsonl(config.deletion_ledger_path, {
        "record_id": result.ledger_record_id,
        "timestamp": stamp,
        "node_id": node_id,
        "node_deleted": result.node_deleted,
        "edges_removed": result.edges_removed,
        "log_entries_removed": result.log_entries_removed,
        "buffer_entries_removed": result.buffer_entries_removed,
        "snapshots_touched": result.snapshots_touched,
        "tombstone_id": result.tombstone_id,
    })

    save_graph(graph, config)
    return result


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_source_refs(refs: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    normalized: list[dict[str, int | str]] = []
    seen: set[tuple[str, int]] = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        session_id = ref.get("session_id")
        if not isinstance(session_id, str):
            continue
        try:
            idx = int(ref.get("message_index"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if (session_id, idx) in seen:
            continue
        seen.add((session_id, idx))
        normalized.append({"session_id": session_id, "message_index": idx})
    return normalized


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _scrub_snapshots(config: AWHMConfig, node_id: str, content: str) -> int:
    """Remove the node (by id or identical content) from every snapshot on disk."""
    snapshots_dir = config.snapshots_dir
    if not snapshots_dir.exists():
        return 0

    target = _normalize_text(content)
    touched = 0
    for snapshot in snapshots_dir.glob("snapshot_*.json"):
        with open(snapshot, encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        graph_data = data.get("graph", {})
        nodes = graph_data.get("nodes", {})
        if isinstance(nodes, dict):
            remove_ids = {
                nid for nid, node in nodes.items()
                if nid == node_id
                or (isinstance(node, dict) and _normalize_text(str(node.get("content", ""))) == target)
            }
            if remove_ids:
                for rid in remove_ids:
                    nodes.pop(rid, None)
                edges = graph_data.get("edges", [])
                if isinstance(edges, list):
                    graph_data["edges"] = [
                        e for e in edges
                        if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                    ]
                changed = True

        wal_state = data.get("wal_state")
        if isinstance(wal_state, list):
            kept = [
                entry for entry in wal_state
                if not (isinstance(entry, dict)
                        and _normalize_text(str(entry.get("content", ""))) == target)
            ]
            if len(kept) != len(wal_state):
                data["wal_state"] = kept
                changed = True

        if changed:
            tmp = snapshot.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(snapshot)
            touched += 1
    return touched
