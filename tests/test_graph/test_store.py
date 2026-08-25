"""Tests for the SQLite graph store."""

from awhm.config import AWHMConfig
from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryEdge, MemoryNode
from awhm.graph.serialization import load_graph, save_graph
from awhm.graph.store import SqliteGraphStore, get_store


def test_sqlite_round_trip(tmp_path, sample_graph):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), storage_backend="sqlite")
    config.ensure_dirs()
    save_graph(sample_graph, config)
    assert config.sqlite_path.exists()
    assert not config.graph_path.exists()

    loaded = load_graph(config)
    assert loaded.node_count() == sample_graph.node_count()
    assert loaded.edge_count() == sample_graph.edge_count()
    for nid, node in sample_graph.nodes.items():
        again = loaded.get_node(nid)
        assert again is not None
        assert again.content == node.content
        assert again.embedding == [float(x) for x in node.embedding] or len(again.embedding) == len(node.embedding)
    assert loaded.get_embedding_matrix()[0].shape == sample_graph.get_embedding_matrix()[0].shape


def test_sqlite_incremental_save(tmp_path, sample_graph):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), storage_backend="sqlite")
    config.ensure_dirs()
    save_graph(sample_graph, config)
    assert sample_graph.consume_changes() == (set(), set(), False)

    sample_graph.add_node(MemoryNode(id="node-4", content="Rust systems programming"))
    sample_graph.get_node("node-1").content = "Python, the programming language"
    sample_graph.mark_dirty("node-1")
    sample_graph.remove_node("node-3")
    sample_graph.add_edge(MemoryEdge(source="node-4", target="node-1"))
    dirty, removed, edges_dirty = sample_graph._dirty_nodes, sample_graph._removed_nodes, sample_graph._edges_dirty
    assert dirty == {"node-4", "node-1"} and removed == {"node-3"} and edges_dirty
    save_graph(sample_graph, config)

    loaded = load_graph(config)
    assert set(loaded.nodes) == {"node-1", "node-2", "node-4"}
    assert loaded.get_node("node-1").content == "Python, the programming language"
    assert loaded.edge_count() == 2


def test_sqlite_imports_existing_json_graph(tmp_path, sample_graph):
    json_config = AWHMConfig(data_dir=str(tmp_path / "awhm"), storage_backend="json")
    json_config.ensure_dirs()
    save_graph(sample_graph, json_config)

    sqlite_config = AWHMConfig(data_dir=str(tmp_path / "awhm"), storage_backend="sqlite")
    loaded = load_graph(sqlite_config)
    assert loaded.node_count() == sample_graph.node_count()
    assert sqlite_config.sqlite_path.exists()


def test_unknown_backend_rejected(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), storage_backend="parquet")
    try:
        get_store(config)
    except ValueError as exc:
        assert "parquet" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_store_direct_empty_load(tmp_path):
    store = SqliteGraphStore(tmp_path / "g.sqlite")
    assert isinstance(store.load(), MemoryGraph)
