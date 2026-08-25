"""Tests for graph serialization."""

import json

from awhm.graph.serialization import load_graph, save_graph


def test_save_and_load(config, sample_graph):
    save_graph(sample_graph, config)
    assert config.graph_path.exists()

    loaded = load_graph(config)
    assert loaded.node_count() == sample_graph.node_count()
    assert loaded.edge_count() == sample_graph.edge_count()

    for nid in sample_graph.nodes:
        assert loaded.get_node(nid) is not None
        assert loaded.get_node(nid).content == sample_graph.get_node(nid).content


def test_load_nonexistent(config):
    graph = load_graph(config)
    assert graph.node_count() == 0


def test_load_legacy_graph_migrates_node_fields(config):
    legacy = {
        "version": "1.0",
        "nodes": {
            "n1": {
                "id": "n1",
                "type": "semantic",
                "content": "My name is Alice",
                "embedding": [],
                "embed_model": "all-MiniLM-L6-v2",
                "strength": {"recency": 1.0, "frequency": 1, "composite": 1.0},
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_accessed": "2024-01-01T00:00:00+00:00",
                "source_sessions": ["s1"],
                "source_refs": [],
                "access_count": 1,
            },
        },
        "edges": [],
    }
    with open(config.graph_path, "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    graph = load_graph(config)
    node = graph.get_node("n1")
    assert node is not None
    assert node.status == "active"
    assert node.valid_from == "2024-01-01T00:00:00+00:00"
