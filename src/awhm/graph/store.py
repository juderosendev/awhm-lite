"""Graph persistence backends.

``json`` writes the whole graph to one file on every save (simple, fine for
thousands of nodes). ``sqlite`` keeps one row per node and only writes rows
that changed since the last save, which is what you want once the graph
holds hundreds of thousands of embeddings.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config import AWHMConfig
from .memory_graph import MemoryGraph
from .models import MemoryEdge, MemoryNode

GRAPH_VERSION = "2.0"


class GraphStore(Protocol):
    def load(self) -> MemoryGraph: ...

    def save(self, graph: MemoryGraph) -> None: ...


# ── JSON ───────────────────────────────────────────────────────


def _migrate_graph_data(data: dict[str, Any]) -> bool:
    """Fill in fields added after v1.0 so old graph files load unchanged."""
    changed = False
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        return changed
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        defaults = {
            "canonical_key": None,
            "status": "active",
            "supersedes": [],
            "valid_from": node.get("created_at"),
            "valid_to": None,
            "confidence": 0.6,
            "entity_type": None,
            "aliases": [],
            "mentioned_dates": [],
        }
        for key, value in defaults.items():
            if key not in node:
                node[key] = value
                changed = True
    return changed


class JsonGraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> MemoryGraph:
        if not self.path.exists():
            return MemoryGraph()
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        data.pop("version", None)
        _migrate_graph_data(data)
        graph = MemoryGraph.from_dict(data)
        graph.consume_changes()
        return graph

    def save(self, graph: MemoryGraph) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"version": GRAPH_VERSION, **graph.to_dict()}
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.path)
        graph.consume_changes()


# ── SQLite ─────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    embedding BLOB,
    embed_dim INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS edges_target ON edges(target);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class SqliteGraphStore:
    """One row per node; saves write only what changed."""

    def __init__(self, path: Path, json_fallback: Path | None = None) -> None:
        self.path = path
        self.json_fallback = json_fallback

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.executescript(_SCHEMA)
        return conn

    def load(self) -> MemoryGraph:
        if not self.path.exists():
            # First run on this backend: import an existing JSON graph, if any.
            if self.json_fallback is not None and self.json_fallback.exists():
                graph = JsonGraphStore(self.json_fallback).load()
                graph.mark_all_dirty()
                self.save(graph)
                return graph
            return MemoryGraph()

        graph = MemoryGraph()
        with self._connect() as conn:
            for node_id, data, blob, dim in conn.execute(
                "SELECT id, data, embedding, embed_dim FROM nodes"
            ):
                node_dict = json.loads(data)
                node_dict["id"] = node_id
                node_dict["embedding"] = (
                    np.frombuffer(blob, dtype=np.float32).tolist() if blob and dim else []
                )
                graph.nodes[node_id] = MemoryNode.from_dict(node_dict)
            for source, target, etype, weight, created_at in conn.execute(
                "SELECT source, target, type, weight, created_at FROM edges"
            ):
                graph.add_edge(MemoryEdge(
                    source=source, target=target, type=etype, weight=weight, created_at=created_at,
                ))
        graph._touch(content_changed=True)
        graph.consume_changes()
        return graph

    def save(self, graph: MemoryGraph) -> None:
        dirty, removed, edges_dirty = graph.consume_changes()
        with self._connect() as conn:
            if removed:
                conn.executemany("DELETE FROM nodes WHERE id = ?", [(nid,) for nid in removed])
            rows = []
            for nid in dirty:
                node = graph.nodes.get(nid)
                if node is None:
                    continue
                payload = node.to_dict()
                embedding = payload.pop("embedding", []) or []
                blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding else None
                rows.append((nid, json.dumps(payload), blob, len(embedding)))
            if rows:
                conn.executemany(
                    "INSERT INTO nodes (id, data, embedding, embed_dim) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET data = excluded.data, "
                    "embedding = excluded.embedding, embed_dim = excluded.embed_dim",
                    rows,
                )
            if edges_dirty:
                conn.execute("DELETE FROM edges")
                conn.executemany(
                    "INSERT INTO edges (source, target, type, weight, created_at) VALUES (?, ?, ?, ?, ?)",
                    [(e.source, e.target, e.type, e.weight, e.created_at) for e in graph.edges],
                )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (GRAPH_VERSION,),
            )
            conn.commit()


def get_store(config: AWHMConfig) -> GraphStore:
    backend = (config.storage_backend or "json").lower()
    if backend == "sqlite":
        return SqliteGraphStore(config.sqlite_path, json_fallback=config.graph_path)
    if backend == "json":
        return JsonGraphStore(config.graph_path)
    raise ValueError(f"Unknown storage_backend {config.storage_backend!r}; use 'json' or 'sqlite'")
