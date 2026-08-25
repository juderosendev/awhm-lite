"""Save/load the memory graph through the configured storage backend."""

from __future__ import annotations

from ..config import AWHMConfig
from .memory_graph import MemoryGraph
from .store import GRAPH_VERSION, _migrate_graph_data, get_store

__all__ = ["GRAPH_VERSION", "_migrate_graph_data", "load_graph", "save_graph"]


def save_graph(graph: MemoryGraph, config: AWHMConfig) -> None:
    """Persist the graph (JSON file or SQLite, per ``config.storage_backend``)."""
    config.ensure_dirs()
    get_store(config).save(graph)


def load_graph(config: AWHMConfig) -> MemoryGraph:
    """Load the graph; returns an empty graph when nothing is stored yet."""
    return get_store(config).load()
