"""Hard deletion: cascade delete across graph, logs, buffer."""

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
    """Hard-delete a node and all associated data for privacy compliance.

    1. Delete node from graph
    2. Sever all incident edges
    3. Delete corresponding raw log entries
    4. Delete from session buffer if present
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

    # Get node info before deletion
    node = graph.get_node(node_id)
    if node is None:
        return result

    content = node.content
    source_sessions = node.source_sessions
    source_refs = _normalize_source_refs(node.source_refs)

    # Count incident edges before deletion.
    result.edges_removed = len(graph.get_edges_for_node(node_id))

    # 1. Remove node (also removes edges)
    removed = graph.remove_node(node_id)
    if removed:
        result.node_deleted = True

    # 3. Delete from raw logs
    reader = RawLogReader(config)
    sessions_affected: set[str] = set()
    total_by_refs = 0
    refs_by_session: dict[str, set[int]] = {}
    for ref in source_refs:
        refs_by_session.setdefault(ref["session_id"], set()).add(ref["message_index"])

    if refs_by_session:
        result.match_strategy = "source_refs"
        for session_id, indices in refs_by_session.items():
            count = reader.hard_delete_entries_by_indices(session_id, indices)
            if count > 0:
                sessions_affected.add(session_id)
            total_by_refs += count
        result.log_entries_removed += total_by_refs
    else:
        result.match_strategy = "exact_content"
        for session_id in source_sessions:
            count = reader.hard_delete_entries_exact(session_id, content)
            if count > 0:
                sessions_affected.add(session_id)
            result.log_entries_removed += count
        result.exact_matches_removed = result.log_entries_removed

    # 4. Delete from buffer
    if buffer:
        before = len(buffer.entries)
        if current_session_id:
            source_msg_ids = {
                ref["message_index"]
                for ref in source_refs
                if ref["session_id"] == current_session_id
            }
        else:
            source_msg_ids = set()

        if source_msg_ids:
            buffer._entries = [
                e for e in buffer._entries
                if e.source_msg not in source_msg_ids
            ]
        else:
            target_norm = _normalize_text(content)
            buffer._entries = [
                e for e in buffer._entries
                if _normalize_text(e.content) != target_norm
            ]
        result.buffer_entries_removed = before - len(buffer._entries)

    result.affected_sessions = sorted(sessions_affected)

    if config.delete_snapshots_on_hard_delete:
        result.snapshots_touched = _scrub_snapshots(config, node_id, content)

    result.tombstone_id = str(uuid.uuid4())
    result.ledger_record_id = str(uuid.uuid4())
    _append_jsonl(
        config.deletion_tombstones_path,
        {
            "tombstone_id": result.tombstone_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": node_id,
            "content_hash_hint": _normalize_text(content)[:64],
            "affected_sessions": result.affected_sessions,
            "match_strategy": result.match_strategy,
        },
    )
    _append_jsonl(
        config.deletion_ledger_path,
        {
            "record_id": result.ledger_record_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": node_id,
            "node_deleted": result.node_deleted,
            "edges_removed": result.edges_removed,
            "log_entries_removed": result.log_entries_removed,
            "buffer_entries_removed": result.buffer_entries_removed,
            "snapshots_touched": result.snapshots_touched,
            "tombstone_id": result.tombstone_id,
        },
    )

    # Save updated graph
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
        msg_idx = ref.get("message_index")
        if not isinstance(session_id, str):
            continue
        try:
            idx = int(msg_idx)
        except (TypeError, ValueError):
            continue
        key = (session_id, idx)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"session_id": session_id, "message_index": idx})
    return normalized


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _scrub_snapshots(config: AWHMConfig, node_id: str, content: str) -> int:
    snapshots_dir = config.snapshots_dir
    if not snapshots_dir.exists():
        return 0

    target_norm = _normalize_text(content)
    touched = 0

    for snapshot in snapshots_dir.glob("snapshot_*.json"):
        with open(snapshot, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        graph_data = data.get("graph", {})
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])

        if isinstance(nodes, dict):
            remove_ids: set[str] = set()
            if node_id in nodes:
                remove_ids.add(node_id)
            for nid, node in list(nodes.items()):
                if not isinstance(node, dict):
                    continue
                if _normalize_text(str(node.get("content", ""))) == target_norm:
                    remove_ids.add(nid)
            if remove_ids:
                for rid in remove_ids:
                    nodes.pop(rid, None)
                if isinstance(edges, list):
                    graph_data["edges"] = [
                        e for e in edges
                        if e.get("source") not in remove_ids and e.get("target") not in remove_ids
                    ]
                changed = True

        wal_state = data.get("wal_state")
        if isinstance(wal_state, list):
            kept_wal = []
            for entry in wal_state:
                if not isinstance(entry, dict):
                    kept_wal.append(entry)
                    continue
                if _normalize_text(str(entry.get("content", ""))) == target_norm:
                    changed = True
                    continue
                kept_wal.append(entry)
            data["wal_state"] = kept_wal

        if changed:
            tmp = snapshot.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(snapshot)
            touched += 1

    return touched
