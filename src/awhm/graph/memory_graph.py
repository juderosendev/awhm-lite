"""MemoryGraph: node/edge CRUD with cached embedding matrix."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .models import MemoryEdge, MemoryNode


class MemoryGraph:
    """Flat directed graph G = (V, E) with embedding matrix caching."""

    def __init__(self) -> None:
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: list[MemoryEdge] = []
        self._embedding_matrix: np.ndarray | None = None
        self._node_id_order: list[str] = []
        self._dirty = True

    # ── Node CRUD ──────────────────────────────────────────────

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node
        self._dirty = True

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> MemoryNode | None:
        node = self.nodes.pop(node_id, None)
        if node is not None:
            self.edges = [
                e for e in self.edges
                if e.source != node_id and e.target != node_id
            ]
            self._dirty = True
        return node

    def update_node_access(self, node_id: str) -> None:
        """Update last_accessed and increment access_count on retrieval hit."""
        node = self.nodes.get(node_id)
        if node:
            node.last_accessed = datetime.now(timezone.utc).isoformat()
            node.access_count += 1

    # ── Edge CRUD ──────────────────────────────────────────────

    def add_edge(self, edge: MemoryEdge) -> None:
        self.edges.append(edge)

    def get_edges_for_node(self, node_id: str) -> list[MemoryEdge]:
        return [e for e in self.edges if e.source == node_id or e.target == node_id]

    def remove_edges_for_node(self, node_id: str) -> int:
        before = len(self.edges)
        self.edges = [
            e for e in self.edges
            if e.source != node_id and e.target != node_id
        ]
        return before - len(self.edges)

    # ── Embedding matrix ───────────────────────────────────────

    def get_embedding_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (matrix, node_ids). Rebuilds if dirty."""
        if self._dirty or self._embedding_matrix is None:
            self._rebuild_embedding_matrix()
        return self._embedding_matrix, self._node_id_order  # type: ignore

    def _rebuild_embedding_matrix(self) -> None:
        ids = []
        embeddings = []
        for nid, node in self.nodes.items():
            if node.embedding:
                ids.append(nid)
                embeddings.append(node.embedding)
        if embeddings:
            self._embedding_matrix = np.array(embeddings, dtype=np.float32)
        else:
            self._embedding_matrix = np.empty((0, 0), dtype=np.float32)
        self._node_id_order = ids
        self._dirty = False

    # ── Queries ────────────────────────────────────────────────

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def all_nodes(self) -> list[MemoryNode]:
        return list(self.nodes.values())

    def all_contents(self) -> list[str]:
        """Return all node contents (for BM25 indexing)."""
        return [n.content for n in self.nodes.values()]

    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    # ── Serialization helpers ──────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryGraph:
        g = cls()
        for nid, nd in data.get("nodes", {}).items():
            g.nodes[nid] = MemoryNode.from_dict(nd)
        for ed in data.get("edges", []):
            g.edges.append(MemoryEdge.from_dict(ed))
        g._dirty = True
        return g
