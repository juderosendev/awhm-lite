"""Atomic JSON save/load with version tag for memory graph."""

from __future__ import annotations

import json
from typing import Any

from ..config import AWHMConfig
from .memory_graph import MemoryGraph

GRAPH_VERSION = "2.0"


def save_graph(graph: MemoryGraph, config: AWHMConfig) -> None:
    """Atomically save graph to JSON (temp file + os.replace)."""
    config.ensure_dirs()
    data: dict[str, Any] = {
        "version": GRAPH_VERSION,
        **graph.to_dict(),
    }
    path = config.graph_path
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def load_graph(config: AWHMConfig) -> MemoryGraph:
    """Load graph from JSON. Returns empty graph if file doesn't exist."""
    path = config.graph_path
    if not path.exists():
        return MemoryGraph()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("version", "1.0")
    _migrate_graph_data(data)
    return MemoryGraph.from_dict(data)


def _migrate_graph_data(data: dict[str, Any]) -> bool:
    """Apply in-memory migrations for backward compatibility."""
    changed = False
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        return changed

    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        created_at = node.get("created_at")
        defaults = {
            "canonical_key": None,
            "status": "active",
            "supersedes": [],
            "valid_from": created_at,
            "valid_to": None,
            "confidence": 0.6,
            "entity_type": None,
        }
        for key, value in defaults.items():
            if key not in node:
                node[key] = value
                changed = True
    return changed
