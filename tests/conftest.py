"""Shared fixtures for all tests."""

from __future__ import annotations

import sys
import os

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from awhm.config import AWHMConfig
from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryEdge, MemoryNode, StrengthScore
from awhm.retrieval.embedding import MockEmbeddingService
from awhm.types import EdgeType, NodeType


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    return str(tmp_path / "awhm_test")


@pytest.fixture
def config(tmp_data_dir):
    """AWHMConfig pointed at tmp directory."""
    cfg = AWHMConfig(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def mock_embedding():
    """Mock embedding service for fast tests."""
    return MockEmbeddingService(dim=384)


@pytest.fixture
def sample_graph(mock_embedding):
    """A graph with a few sample nodes and edges."""
    g = MemoryGraph()

    embs = mock_embedding.encode([
        "Python programming language",
        "User prefers dark mode",
        "Deploy to production on Fridays",
    ])

    n1 = MemoryNode(
        id="node-1",
        type=NodeType.SEMANTIC.value,
        content="Python programming language",
        embedding=embs[0].tolist(),
        source_sessions=["session-1"],
    )
    n2 = MemoryNode(
        id="node-2",
        type=NodeType.PROCEDURAL.value,
        content="User prefers dark mode",
        embedding=embs[1].tolist(),
        source_sessions=["session-1"],
    )
    n3 = MemoryNode(
        id="node-3",
        type=NodeType.EPISODIC.value,
        content="Deploy to production on Fridays",
        embedding=embs[2].tolist(),
        source_sessions=["session-2"],
    )

    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    e1 = MemoryEdge(
        source="node-1", target="node-2",
        type=EdgeType.ASSOCIATION.value, weight=0.8,
    )
    g.add_edge(e1)

    return g
