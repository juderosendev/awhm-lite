"""Tests for MemoryGraph."""

from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryEdge, MemoryNode
from awhm.types import EdgeType, NodeType


def test_add_and_get_node():
    g = MemoryGraph()
    node = MemoryNode(id="n1", content="test", type=NodeType.SEMANTIC.value)
    g.add_node(node)
    assert g.get_node("n1") is not None
    assert g.node_count() == 1


def test_remove_node_severs_edges():
    g = MemoryGraph()
    g.add_node(MemoryNode(id="n1", content="a"))
    g.add_node(MemoryNode(id="n2", content="b"))
    g.add_edge(MemoryEdge(source="n1", target="n2", type=EdgeType.ASSOCIATION.value))

    g.remove_node("n1")
    assert g.get_node("n1") is None
    assert g.edge_count() == 0


def test_embedding_matrix(mock_embedding):
    g = MemoryGraph()
    embs = mock_embedding.encode(["hello", "world"])
    g.add_node(MemoryNode(id="n1", content="hello", embedding=embs[0].tolist()))
    g.add_node(MemoryNode(id="n2", content="world", embedding=embs[1].tolist()))

    matrix, ids = g.get_embedding_matrix()
    assert matrix.shape == (2, 384)
    assert set(ids) == {"n1", "n2"}


def test_update_node_access():
    g = MemoryGraph()
    node = MemoryNode(id="n1", content="test", access_count=1)
    g.add_node(node)
    g.update_node_access("n1")
    assert g.get_node("n1").access_count == 2


def test_to_from_dict(sample_graph):
    data = sample_graph.to_dict()
    g2 = MemoryGraph.from_dict(data)
    assert g2.node_count() == sample_graph.node_count()
    assert g2.edge_count() == sample_graph.edge_count()
