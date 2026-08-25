"""MemoryGraph: node/edge storage with an adjacency index and cached embeddings."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..types import NodeStatus
from .models import MemoryEdge, MemoryNode


class MemoryGraph:
    """Flat directed graph G = (V, E).

    Two cheap change counters let callers cache derived structures:

    * ``content_version`` changes when nodes are added or removed, i.e. when
      anything that affects lexical or embedding indexes changes.
    * ``version`` changes on every mutation, including edges and access
      bookkeeping.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: list[MemoryEdge] = []
        self._edges_by_node: dict[str, list[MemoryEdge]] = defaultdict(list)
        self._embedding_matrix: np.ndarray | None = None
        self._node_id_order: list[str] = []
        self._dirty = True
        self.version = 0
        self.content_version = 0

    # ── Change tracking ────────────────────────────────────────

    def _touch(self, content_changed: bool = False) -> None:
        self.version += 1
        if content_changed:
            self.content_version += 1
            self._dirty = True

    # ── Node CRUD ──────────────────────────────────────────────

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node
        self._touch(content_changed=True)

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> MemoryNode | None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return None
        self.remove_edges_for_node(node_id)
        self._touch(content_changed=True)
        return node

    def update_node_access(self, node_id: str, now: datetime | None = None) -> None:
        """Record a retrieval hit: bump ``last_accessed`` and ``access_count``."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        node.last_accessed = stamp
        node.access_count += 1
        self._touch()

    # ── Edge CRUD ──────────────────────────────────────────────

    def add_edge(self, edge: MemoryEdge) -> None:
        self.edges.append(edge)
        self._edges_by_node[edge.source].append(edge)
        if edge.target != edge.source:
            self._edges_by_node[edge.target].append(edge)
        self._touch()

    def get_edges_for_node(self, node_id: str) -> list[MemoryEdge]:
        return list(self._edges_by_node.get(node_id, ()))

    def remove_edges_for_node(self, node_id: str) -> int:
        incident = self._edges_by_node.pop(node_id, [])
        if not incident:
            return 0
        doomed = set(map(id, incident))
        self.edges = [e for e in self.edges if id(e) not in doomed]
        for edge in incident:
            other = edge.target if edge.source == node_id else edge.source
            if other != node_id:
                self._edges_by_node[other] = [
                    e for e in self._edges_by_node.get(other, []) if id(e) not in doomed
                ]
        self._touch()
        return len(incident)

    # ── Embedding matrix ───────────────────────────────────────

    def get_embedding_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return ``(matrix, node_ids)``, rebuilding only when nodes changed."""
        if self._dirty or self._embedding_matrix is None:
            self._rebuild_embedding_matrix()
        assert self._embedding_matrix is not None
        return self._embedding_matrix, self._node_id_order

    def _rebuild_embedding_matrix(self) -> None:
        ids: list[str] = []
        embeddings: list[list[float]] = []
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

    def active_nodes(self) -> list[MemoryNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE.value]

    def all_contents(self) -> list[str]:
        """Return all node contents in ``node_ids()`` order (for BM25 indexing)."""
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
            g.add_edge(MemoryEdge.from_dict(ed))
        g._touch(content_changed=True)
        return g
